

import subprocess as sp


def kwargs_to_cmd_args(kwargs):
    args = []
    for k, v in kwargs.items():
        args.append('-%s' % k)
        if v is not None:
            args.append('%s' % str(v))
    return args


def _check_input(input_fn: str, input_data: bytes):
    capture_stdin = (input_fn == 'pipe:')
    if capture_stdin and input_data is None:
        raise ValueError(
            "the buffered video data for stdin should not be empty")


def run_command_async(cmd_args,
                      pipe_stdin=True,
                      pipe_stdout=False,
                      pipe_stderr=False,
                      quiet=False):
    stdin_stream = sp.PIPE if pipe_stdin else None
    stdout_stream = sp.PIPE if pipe_stdout or quiet else None
    stderr_stream = sp.PIPE if pipe_stderr or quiet else None

    return sp.Popen(
        cmd_args,
        stdin=stdin_stream,
        stdout=stdout_stream,
        stderr=stderr_stream,
        close_fds=True)


def wait(process):
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return (output, rc)


def run_command(cmd_args,
                input=None,
                pipe_stdin=True,
                pipe_stdout=False,
                pipe_stderr=False,
                quiet=False):
    with run_command_async(
            cmd_args,
            pipe_stdin=pipe_stdin,
            pipe_stdout=pipe_stdout,
            pipe_stderr=pipe_stderr,
            quiet=quiet) as proc:
        stdout, stderr = proc.communicate(input)
        retcode = proc.poll()

        if retcode:
            raise Exception('ffmpeg error: %s' % stderr)

        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

        return stdout, stderr
