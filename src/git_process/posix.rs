//! Isolate child descriptors and coordinate native spawn with HDF5 close/open.
use std::ffi::{CString, OsStr};
use std::fs::File;
use std::io::{self, Read};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;

fn native_result(code: libc::c_int) -> io::Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(code))
    }
}

struct Attributes(libc::posix_spawnattr_t);
impl Attributes {
    fn new() -> io::Result<Self> {
        let mut value = MaybeUninit::uninit();
        // SAFETY: init writes the opaque handle; it is owned only after success.
        unsafe {
            native_result(libc::posix_spawnattr_init(value.as_mut_ptr()))?;
            Ok(Self(value.assume_init()))
        }
    }
}
impl Drop for Attributes {
    fn drop(&mut self) {
        // SAFETY: this handle was successfully initialized and has one owner.
        unsafe {
            libc::posix_spawnattr_destroy(&mut self.0);
        }
    }
}

struct Actions(libc::posix_spawn_file_actions_t);
impl Actions {
    fn new() -> io::Result<Self> {
        let mut value = MaybeUninit::uninit();
        // SAFETY: init writes the opaque handle; it is owned only after success.
        unsafe {
            native_result(libc::posix_spawn_file_actions_init(value.as_mut_ptr()))?;
            Ok(Self(value.assume_init()))
        }
    }
}
impl Drop for Actions {
    fn drop(&mut self) {
        // SAFETY: this handle was successfully initialized and has one owner.
        unsafe {
            libc::posix_spawn_file_actions_destroy(&mut self.0);
        }
    }
}

fn above_stdio(fd: OwnedFd) -> io::Result<OwnedFd> {
    if fd.as_raw_fd() > libc::STDERR_FILENO {
        return Ok(fd);
    }
    // SAFETY: duplicate our owned pipe end to avoid file-action collisions if the
    // parent has closed a standard descriptor. No pre-existing parent FD is changed.
    let duplicate = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
    if duplicate < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fcntl returned a new, uniquely owned descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(duplicate) })
}

pub(super) fn stdout(program: &OsStr, args: &[&OsStr]) -> io::Result<Option<Vec<u8>>> {
    let program = CString::new(program.as_bytes())?;
    let mut arguments = Vec::with_capacity(args.len() + 1);
    arguments.push(program);
    for arg in args {
        arguments.push(CString::new(arg.as_bytes())?);
    }
    let environment: Vec<CString> = std::env::vars_os()
        .map(|(key, value)| {
            let mut entry = key.as_bytes().to_vec();
            entry.push(b'=');
            entry.extend_from_slice(value.as_bytes());
            CString::new(entry)
        })
        .collect::<Result<_, _>>()?;
    // posix_spawnp uses mutable pointer types but does not modify these owned strings.
    let mut argv: Vec<_> = arguments.iter().map(|s| s.as_ptr().cast_mut()).collect();
    argv.push(std::ptr::null_mut());
    let mut envp: Vec<_> = environment.iter().map(|s| s.as_ptr().cast_mut()).collect();
    envp.push(std::ptr::null_mut());

    let (reader, writer) = io::pipe()?;
    let reader = above_stdio(reader.into())?;
    let writer = above_stdio(writer.into())?;
    let mut actions = Actions::new()?;
    let mut attributes = Attributes::new()?;
    let mut pid = 0;
    // SAFETY: initialized handles, valid owned FDs and NUL-terminated arrays all
    // remain alive through spawn. Only the three standard descriptors are retained.
    unsafe {
        native_result(libc::posix_spawn_file_actions_addopen(
            &mut actions.0,
            libc::STDIN_FILENO,
            c"/dev/null".as_ptr(),
            libc::O_RDONLY,
            0,
        ))?;
        native_result(libc::posix_spawn_file_actions_addopen(
            &mut actions.0,
            libc::STDERR_FILENO,
            c"/dev/null".as_ptr(),
            libc::O_WRONLY,
            0,
        ))?;
        native_result(libc::posix_spawn_file_actions_adddup2(
            &mut actions.0,
            writer.as_raw_fd(),
            libc::STDOUT_FILENO,
        ))?;
        #[cfg(target_os = "macos")]
        native_result(libc::posix_spawn_file_actions_addclose(
            &mut actions.0,
            writer.as_raw_fd(),
        ))?;
        let mut defaults = MaybeUninit::uninit();
        if libc::sigemptyset(defaults.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error());
        }
        let mut defaults = defaults.assume_init();
        if libc::sigaddset(&mut defaults, libc::SIGPIPE) != 0 {
            return Err(io::Error::last_os_error());
        }
        native_result(libc::posix_spawnattr_setsigdefault(
            &mut attributes.0,
            &defaults,
        ))?;
        let flags = libc::POSIX_SPAWN_SETSIGDEF as i32;
        #[cfg(target_os = "macos")]
        let flags = flags | libc::POSIX_SPAWN_CLOEXEC_DEFAULT;
        native_result(libc::posix_spawnattr_setflags(
            &mut attributes.0,
            flags as i16,
        ))?;
        #[cfg(test)]
        super::tests::configure_spawn_file_actions(&mut actions.0)?;
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        // GNU libc >=2.34 closes the child's descriptors after stdout is duplicated.
        // Keep this after the test gate so the regression observes the inherited
        // descriptors during spawn, before the close-from action has run.
        native_result(libc::posix_spawn_file_actions_addclosefrom_np(
            &mut actions.0,
            3,
        ))?;
        // Native spawn temporarily retains the parent's open file descriptions.
        // Coordinate that interval with HDF5 close/open, even when exec closes FDs.
        // Release this existing recursive HDF5 mutex before pipe draining/waiting.
        native_result(hdf5_metno::sync::sync(|| {
            libc::posix_spawnp(
                &mut pid,
                arguments[0].as_ptr(),
                &actions.0,
                &attributes.0,
                argv.as_ptr(),
                envp.as_ptr(),
            )
        }))?;
    }
    drop(writer);
    let mut reader = File::from(reader);
    let mut bytes = Vec::new();
    let read_result = reader.read_to_end(&mut bytes);
    // On read failure, closing the pipe lets writers receive SIGPIPE rather than
    // block while we reap. Always wait after a successful spawn, including errors.
    drop(reader);
    let mut status = 0;
    loop {
        // SAFETY: pid names our successfully spawned child; status is writable.
        if unsafe { libc::waitpid(pid, &mut status, 0) } >= 0 {
            break;
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
    read_result?;
    Ok((libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0).then_some(bytes))
}
