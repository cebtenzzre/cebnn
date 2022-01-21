#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import errno
import fcntl
import io
import locale
import os
import pathlib
import re
import select
import shlex
import stat
import subprocess
import sys
import threading
from collections import defaultdict
from enum import Enum
from itertools import chain, count
from queue import Queue
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Union

T = TypeVar('T')


ALL_SUBDIR = 'orig'
CLASS_SUBDIR = 'class'
WEIGHT_SUBDIR = 'weight'
TEST_SUBDIR = 'test'
TEST_ONLY_SUBDIR = 'test_only'

WEIGHT_REGEX = re.compile('weight=([0-9.]+)')


def dolink(dest: str, srcf: str, dest_fn: str) -> None:
    dstf = os.path.join(dest, dest_fn)
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)

    try:
        os.symlink(srcf, dstf)
    except FileExistsError:
        pass  # Link already exists


def readwrite(infile: io.FileIO, outfile: io.FileIO, input_lines: Iterable[str]) -> Iterable[str]:
    from select import POLLERR, POLLHUP, POLLIN, POLLNVAL, POLLOUT

    encoding = locale.getpreferredencoding(False)
    assert '\n'.encode(encoding) == b'\n', 'Unsupported system encoding'

    infd  = infile.fileno()
    outfd = outfile.fileno()
    input_iterator = iter(input_lines)
    read_buffer  = bytearray()
    write_buffer = bytearray()

    # Set fds non-blocking
    for fd in (infd, outfd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    poll = select.poll()
    pollfds = {infd, outfd}
    poll.register(infd, POLLIN)
    poll.register(outfd, POLLOUT)

    while pollfds:
        poll_result = poll.poll()
        if not poll_result:
            continue  # Wait for an event
        for fd, events in poll_result:
            if events & (POLLERR | POLLNVAL):
                pollfds.remove(fd)
                poll.unregister(fd)
                if fd == infd:
                    yield read_buffer.decode(encoding)  # Flush read buffer
                elif fd == outfd:
                    outfile.close()
                    write_buffer = bytearray()  # Clear write buffer
        fd_revents = {infd: 0, outfd: 0}
        fd_revents.update(poll_result)

        if fd_revents[infd] & POLLHUP and not fd_revents[infd] & POLLIN:
            pollfds.remove(infd)
            poll.unregister(infd)
            yield read_buffer.decode(encoding)  # Flush read buffer
        if fd_revents[outfd] & POLLHUP:
            pollfds.remove(outfd)
            poll.unregister(outfd)
            outfile.close()
            write_buffer = bytearray()  # Clear write buffer

        # Try to read from infd
        if fd_revents[infd] & POLLIN:
            while True:
                old_buflen = len(read_buffer)
                read_bytes = infile.read()
                if read_bytes is None:
                    break  # No bytes available
                if not read_bytes:
                    # No more data to read, dispose of our input fd
                    pollfds.remove(infd)
                    poll.unregister(infd)
                    infile.close()
                    break  # Out of data
                read_buffer.extend(read_bytes)
                if b'\n' in read_buffer[old_buflen:]:
                    while True:
                        line, sep, tmp = read_buffer.partition(b'\n')
                        if not sep:
                            break
                        read_buffer = tmp
                        yield line.decode(encoding)
                # Keep reading until a read would block

        # Try to write to outfd
        if fd_revents[outfd] & POLLOUT:
            while True:
                if not write_buffer:
                    try:
                        out_line = next(input_iterator)
                    except StopIteration:
                        # No more data to write, dispose of our output fd
                        pollfds.remove(outfd)
                        poll.unregister(outfd)
                        outfile.close()
                        break  # Out of data
                    else:
                        write_buffer.extend((out_line + '\n').encode(encoding))
                        assert write_buffer
                bytes_written = outfile.write(write_buffer)
                if bytes_written is None:
                    break  # No space available
                write_buffer = write_buffer[bytes_written:]


orig_files: Dict[str, str] = {}


def process_image(f: str, norm_ext: str) -> None:
    fname = os.path.basename(f)
    dest_fname = '{}.{}'.format(os.path.splitext(fname)[0], norm_ext)
    src_file = os.path.abspath(f)
    assert os.path.exists(src_file)

    dest = os.path.join(dest_dir, ALL_SUBDIR)
    dupe_found = False  # We need this in case a file is found again for a different class

    def files_match(filea: str, fileb: str) -> bool:
        filea = os.path.realpath(filea)
        fileb = os.path.realpath(fileb)
        if os.stat(filea).st_size != os.stat(fileb).st_size:
            return False  # No match (different size)
        args = ('rmlint', '--is-reflink', '-V', '--', filea, fileb)
        s = subprocess.run(args).returncode
        if s in (0, 6, 7, 8):
            return True  # Matching files (without reading data)
        elif s not in (5, 10, 11):
            raise RuntimeError('Unexpected rmlint status: {}\nCommand: {}'.format(s, args))
        # Inconclusive status, compare content
        s = subprocess.run(('cmp', '-s', '--', filea, fileb)).returncode
        if s not in (0, 1):
            raise RuntimeError('Unexpected cmp status: {}'.format(s))
        return s == 0

    # Find an available name
    for num in count(start=1):
        dstf_noext = os.path.splitext(dest_fname)[0]
        if (conflict_ext := orig_files.get(dstf_noext)) is None:
            break  # Found one!
        conflict_fname = dstf_noext + conflict_ext
        if files_match(src_file, os.path.join(dest, conflict_fname)):
            dest_fname = conflict_fname
            dupe_found = True
            break

        # Legitimate conflict, rename it
        dest_fname = '{} ({}).{}'.format(os.path.splitext(fname)[0], num, norm_ext)

    # Link it!
    if not dupe_found:
        base, ext = os.path.splitext(dest_fname)
        orig_files[base] = ext
        dolink(os.path.join(dest_dir, ALL_SUBDIR),
               os.path.realpath(src_file),
               dest_fname)
    for c in src_classes:
        dolink(os.path.join(dest_dir, CLASS_SUBDIR, c),
               os.path.join('..', '..', ALL_SUBDIR, dest_fname),
               dest_fname)
    dolink(os.path.join(dest_dir, WEIGHT_SUBDIR, src_weight),
           os.path.join('..', '..', ALL_SUBDIR, dest_fname),
           dest_fname)
    if src_test:
        dolink(os.path.join(dest_dir, TEST_SUBDIR),
               os.path.join('..', ALL_SUBDIR, dest_fname),
               dest_fname)
    if src_test_only:
        dolink(os.path.join(dest_dir, TEST_ONLY_SUBDIR),
               os.path.join('..', ALL_SUBDIR, dest_fname),
               dest_fname)


def get_files(top: str, depth: int) -> Iterator[os.DirEntry[str]]:
    def getfs(d: Union[str, os.DirEntry[str]], dirs: bool) -> Iterator[os.DirEntry[str]]:
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) == dirs:
                    yield entry

    dirs: Iterable[Union[str, os.DirEntry[str]]] = (top,)
    for _ in range(depth - 1):
        dirs = chain.from_iterable(getfs(d, dirs=True) for d in dirs)
    yield from chain.from_iterable(getfs(d, dirs=False) for d in dirs)


def get_files_r(top: str, followlinks: bool = False, maxdepth: Optional[int] = None) -> Iterator[str]:
    def numparts(path: str) -> int:
        return len(pathlib.Path(path).parts)
    top_parts = numparts(top)
    for root, dirs, files in os.walk(top, followlinks=followlinks):
        if maxdepth is not None:
            root_parts = numparts(root) + 1  # +1 because we're inside of root
            depth = root_parts - top_parts  # Depth of the yielded file paths
            assert depth <= maxdepth
            if depth == maxdepth:
                dirs.clear()  # Do not descend past this depth
        # Yield (dirname, path) pairs
        yield from (os.path.join(root, f) for f in files)


class BISentinel(Enum):
    _QUIT = 0


class BufferedIterator(Generic[T]):
    def __init__(self, it: Iterator[T], name: Optional[str] = None, maxsize: int = 1024) -> None:
        self.it = it
        self.queue: Queue[Any] = Queue(maxsize=maxsize)  # Queue[Union[T, BISentinel]]
        self.thread = threading.Thread(target=self.run_thread, name=name, daemon=True)
        self.thread.start()

    def __iter__(self) -> Iterator[T]:
        while True:
            item = self.queue.get()
            if item is BISentinel._QUIT:
                break  # End of data
            yield item
        self.thread.join()

    def run_thread(self) -> None:
        try:
            for item in self.it:
                self.queue.put(item)
        finally:
            self.queue.put(BISentinel._QUIT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='%(prog)s [-L] [--] dest_dir [find depth]:[class]:source_dir ...')
    parser.add_argument('-L', action='store_true', dest='follow_links', help='Follow symlinks')
    parser.add_argument('dest_dir')
    parser.add_argument('sources', nargs='+')
    options = parser.parse_args()

    # Namespace is untyped; explicitly type its fields
    follow_links: bool = options.follow_links
    dest_dir: str = options.dest_dir
    sources: List[str] = options.sources
    del parser, options

    try:
        os.mkdir(dest_dir)
    except FileExistsError:
        pass  # Directory already exists

    # This is important because of the way we find conflicts
    if os.path.exists(os.path.join(dest_dir, ALL_SUBDIR)):
        print('Error: Output directory not empty.', file=sys.stderr)
        sys.exit(1)

    # Limit searched files to those that seem like media
    MEDIA_EXTS = frozenset((
        'bmp', 'exr', 'gif', 'gifv', 'jpg', 'jpeg', 'pbm', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'))

    for src in sources:
        src_find_depth_s, src_class, src_path = src.split(':')
        src_find_depth = int(src_find_depth_s) if src_find_depth_s else None

        # Parse classes
        src_classes = []
        src_weight = '1'
        src_test = False
        src_test_only = False
        if src_class:
            for c in src_class.split(','):
                if (m := WEIGHT_REGEX.fullmatch(c)):
                    src_weight = m.group(1)
                elif c == 'test':
                    src_test = True
                elif c == 'test_only':
                    src_test_only = True
                else:
                    src_classes.append(c)
        del src_class

        print('Reading {}...'.format(src_path))

        args = ('parallel', '-rd', '\n', '-n1',
                r'ext=$(./get_image_ext.sh {}); if [[ -n $ext ]]; then printf "%s\n%s\n" {} "$ext"; fi')
        parallelproc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        assert isinstance(parallelproc.stdin,  io.FileIO)
        assert isinstance(parallelproc.stdout, io.FileIO)
        try:
            def get_paths() -> Iterator[str]:
                for path in get_files_r(src_path, followlinks=follow_links, maxdepth=src_find_depth):
                    if path.startswith('.'):
                        continue  # Hidden file
                    if os.path.splitext(path)[1][1:] not in MEDIA_EXTS:
                        continue  # Does not look like media
                    try:
                        if not stat.S_ISREG(os.stat(path).st_mode):
                            continue  # Not a regular file
                    except FileNotFoundError:
                        continue  # Broken link
                    yield path
            gpbuf = BufferedIterator(get_paths(), name='get_paths')
            gei_lines = readwrite(parallelproc.stdout, parallelproc.stdin, gpbuf)
            for f, norm_ext in zip(gei_lines, gei_lines):
                process_image(f, norm_ext)
            parallelproc.wait()
        except:
            parallelproc.terminate()
            parallelproc.wait()
            raise

    all_dir = os.path.join(dest_dir, ALL_SUBDIR)
    class_dir = os.path.join(dest_dir, CLASS_SUBDIR)
    weight_dir = os.path.join(dest_dir, WEIGHT_SUBDIR)
    test_dir = os.path.join(dest_dir, TEST_SUBDIR)
    test_only_dir = os.path.join(dest_dir, TEST_ONLY_SUBDIR)

    # Clean up with rmlint (NB: requires patch: "Make follow_symlinks do what I would expect")
    outfd_rd, outfd_wr = os.pipe()
    try:
        cmdline = ('rmlint -o sh:stdout -VVV -fx -T df -S \'X< \\([0-9]+\\)\\.[^ ]+$>ma\''
                   r' -c sh:cmd="printf >&{} ' r"'%s\\n%s\\n'" r' \"\$1\" \"\$2\"" {} | bash -s -- -dxpq'
                   .format(outfd_wr, shlex.quote(all_dir)))
        rmlintproc = subprocess.Popen(cmdline, shell=True, text=True, stdout=sys.stderr, pass_fds=(outfd_wr,))
    except:
        os.close(outfd_rd)
        raise
    finally:
        os.close(outfd_wr)

    dirs = [(class_dir, 2)]
    if os.path.exists(weight_dir):
        dirs.append((weight_dir, 2))
    if os.path.exists(test_dir):
        dirs.append((test_dir, 1))
    if os.path.exists(test_only_dir):
        dirs.append((test_only_dir, 1))

    link_lookup: DefaultDict[str, List[str]] = defaultdict(list)
    for entry in chain.from_iterable(get_files(d, depth=depth) for d, depth in dirs):
        try:
            lname = os.readlink(entry)
        except OSError as e:
            if e.errno != errno.EINVAL:
                raise
            continue  # Not a symlink
        link_lookup[os.path.basename(lname)].append(entry.path)

    try:
        with open(outfd_rd) as child_out:
            paths = (l.rstrip('\n') for l in child_out)
            for inferior, replacement in zip(paths, paths):
                bad_fn = os.path.basename(inferior)
                good_fn = os.path.basename(replacement)
                if (links := link_lookup.get(bad_fn)) is not None:
                    for link in links:
                        os.unlink(link)
                        dest = os.path.dirname(link)
                        os.symlink(os.path.relpath(os.path.join(all_dir, good_fn), start=dest),
                                   os.path.join(dest, os.path.basename(link)))
                    del link_lookup[bad_fn]  # Remove the deleted links
                os.unlink(inferior)
        rmlintproc.wait()
    except:
        rmlintproc.terminate()
        rmlintproc.wait()
        raise

    for f in get_files_r(class_dir):
        if not os.path.exists(os.path.realpath(f)):
            print('Found dead link: {}'.format(f))
