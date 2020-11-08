import json
import subprocess


def decodeCmd(cmd, sepBy):
    cmd = [cmd.strip() for cmd in cmd.split('\n')]
    cmd = [cmd for cmd in cmd if cmd and not cmd.startswith('#')]
    cmd = sepBy.join(cmd)
    return cmd


class Error(Exception):
    def __init__(self, cmd, stdout, stderr):
        super(Error, self).__init__(f'{cmd} error (see stderr output for detail)')
        self.stdout = stdout
        self.stderr = stderr


def exeIt(cmd, returnStdout=True, returnStderr=True, input=None, sepBy=' ', debug=False):
    pipe_stdin = None  # implement streaming input
    stdin_stream = subprocess.PIPE if pipe_stdin else None
    stdout_stream = subprocess.PIPE if returnStdout else None
    stderr_stream = subprocess.PIPE if returnStderr else None
    cmd = decodeCmd(cmd, sepBy)
    if debug:
        print(f"\n{'_' * 100}\nbash cmd: {cmd}\n{'_' * 100}\n")
    process = subprocess.Popen(cmd, shell=True, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream)
    out, err = process.communicate(input)
    retcode = process.poll()
    # if retcode:
    #     raise Error('ffmpeg', out, err)
    if out is not None:
        out = out.decode()
    if err is not None:
        err = err.decode()
    return retcode, out, err


def curlIt(data, host='', port='', call='', url='', method='POST', timeout=60, debug=False):
    if not url:
        url = f'{host}:{port}'
    if call:
        url = f'{url}/{call}'
    curlCmd = f"curl -X {method} '{url}' -d '{json.dumps(data)}' -m {timeout}"
    return exeIt(cmd=curlCmd, sepBy='', debug=debug)
