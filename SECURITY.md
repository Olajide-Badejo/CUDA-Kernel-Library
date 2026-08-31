# Security Policy

## What this project is

CUDA Kernel Library is a research and portfolio project: a single author library
built to study GPU kernel optimization, with the measurements and the reasoning
published alongside the code. It is not a hardened production dependency, and I
am not running a security release process behind it.

Say so plainly, because the alternative is a policy nobody can keep:

- There is no long term support branch and no backport program.
- There are no supported security releases. A fix lands on `main` and goes out
  with the next version, whenever that is.
- I make no commitment to a response time.

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.1.x   | Fixes land on `main` and ship in the next release. |
| earlier | Not supported. |

## Reporting a vulnerability

Open an issue at
<https://github.com/Olajide-Badejo/CUDA-Kernel-Library/issues>. If the details
should not be public, open an issue that says only that you have a security
report and I will arrange a private channel through GitHub. GitHub private
vulnerability reporting is also enabled on the repository, and that is the
better route when the finding is genuinely sensitive.

Useful things to include: the version or commit, the CUDA toolkit and driver,
the GPU architecture, and the smallest input that shows the problem.

## Threat model, so a report is not wasted

The library takes device pointers, shapes and leading dimensions from the caller
and hands them to CUDA kernels. It validates dimensions, leading dimensions and
null pointers, and it refuses shapes a path cannot take. It does not and cannot
validate that a device pointer actually addresses an allocation large enough for
the shape you described, so a caller who lies about the shape gets an out of
bounds device access. That is the same contract BLAS has always had, and it is
not a vulnerability in the library.

What I do want to hear about: a shape the validation accepts that then reads or
writes outside the buffers the descriptor implies, an input that makes the C ABI
leak or double free, or anything that makes an entry point throw across the C
boundary instead of returning a status.
