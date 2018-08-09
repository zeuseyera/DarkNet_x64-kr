/* Declarations for getopt.
   Copyright (C) 1989, 90, 91, 92, 93, 94 Free Software Foundation, Inc.

This file is part of the GNU C Library.  Its master source is NOT part of
the C library, however.  The master source lives in /gd/gnu/lib.

The GNU C Library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.

The GNU C Library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with the GNU C Library; see the file COPYING.LIB.  If
not, write to the Free Software Foundation, Inc., 675 Mass Ave,
Cambridge, MA 02139, USA.  */

#ifndef _GETOPT_H
#define _GETOPT_H 1

#ifdef	__cplusplus
extern "C" {
#endif

/* For communication from `getopt' to the caller.
   When `getopt' finds an option that takes an argument,
   the argument value is returned here.
   Also, when `ordering' is RETURN_IN_ORDER,
   each non-option ARGV-element is returned here.  */
/* 'getopt'���� ȣ���ڷ� ����� ���Ͽ�.
   'getopt'�� ������ ���� ������ ã����, ���� ���� ���⿡ ��ȯ�Ѵ�.
   ����, 'odering'�� RETURN_IN_ORDER�̴�, ������ ��-���� ARGV-��Ҵ� ���⿡ ��ȯ�� ���̴�. */
extern char *optarg;

/* Index in ARGV of the next element to be scanned.
   This is used for communication to and from the caller
   and for communication between successive calls to `getopt'.

   On entry to `getopt', zero means this is the first call; initialize.

   When `getopt' returns EOF, this is the index of the first of the
   non-option elements that the caller should itself scan.

   Otherwise, `optind' communicates from one call to the next
   how much of ARGV has been scanned so far.  */
/* ARGV ���� �f������ ���� ����� ����.
   �̰��� ȣ���ڿ��� ��ſ� ���ȴ� �׸��� 'getopt'�� ���� �������� ȣ�� ������ ��ſ� ���ȴ�.
   
   'getopt' �� ����, 0 �� ù��° ȣ���� �ǹ��Ѵ�; �ʱ�ȭ �Ѵ�.
   
   'getopt' �� EOF �� ��ȯ�� ��, �̰��� ȣ���ڰ� ������ �f����ϴ� ��-���� ����� ù��° �����̴�.
   
   �׷��� ������, 'optind' �� ARGV�� ���ݱ��� �󸶳� ���� �f������
   �ϳ��� ȣ�⿡�� ���� ȣ��� ����Ѵ�. */
extern int optind;

/* Callers store zero here to inhibit the error message `getopt' prints
   for unrecognized options.  */
/* ȣ���ڴ� �ν��� �� ���� �����׿� ���Ͽ� ���� �޽����� �����ϱ� ���Ͽ� `getopt' ��¿� 0�� �����Ѵ�.*/
extern int opterr;

/* Set to an option character which was unrecognized.  */
/* �ν��� �� ���� �϶� ������ ���ڷ� �����Ѵ�.*/
extern int optopt;

/* Describe the long-named options requested by the application.
   The LONG_OPTIONS argument to getopt_long or getopt_long_only is a vector
   of `struct option' terminated by an element containing a name which is
   zero.

   The field `has_arg' is:
   no_argument		(or 0) if the option does not take an argument,
   required_argument	(or 1) if the option requires an argument,
   optional_argument 	(or 2) if the option takes an optional argument.

   If the field `flag' is not NULL, it points to a variable that is set
   to the value given in the field `val' when the option is found, but
   left unchanged if the option is not found.

   To have a long-named option do something other than set an `int' to
   a compiled-in constant, such as set a value from `optarg', set the
   option's `flag' field to zero and its `val' field to a nonzero
   value (the equivalent single-letter option character, if there is
   one).  For long options that have a zero `flag' field, `getopt'
   returns the contents of the `val' field.  */
/* �������α׷����� ��û�� �� �̸� �������� �����Ѵ�.
   getopt_long �Ǵ� getopt_long_only �� LONG_OPTIONS ������ �̸��� 0 �� �����ϴ� ��ҷ�
   ����� 'option ����' �� �����̴�.
   
   'has_arg' �׸�:
   no_argument			(�Ǵ� 0): �������� ������ ������ ���� ���
   required_argument	(�Ǵ� 1): �������� ������ �ʿ��� ���
   optional_argument	(�Ǵ� 2): �������� ������ ������ ���� ���
   
   'flag' �׸��� NULL �� �ƴ� ���, �������� �߰ߵǸ� 'val' �׸� �־��� ������ �����ϰ�
   �� ������ ����Ų��, ������ �������� ã�� �� ���� ���� ������� �ʴ´�.
   
   long-named �������� ����Ϸ��� compiled-in ����� 'int'�� �����ϰ�, 'optarg'���� ���� �����ϴ�
   �Ͱ� ����, �����׵��� 'flag' �׸��� 0 ���� �׸��� 'val' �׸��� 0 �� �ƴ� ������ �����Ѵ�
   (������ ����-���� ������ ����, �ϳ��� �ִٸ�). �� �����׿� ���Ͽ� 'flag' �׸��� 0 �� ���,
   'getopt' �� 'val' �׸��� ������ ��ȯ�Ѵ�. */
struct option
{
#if defined (__STDC__) && __STDC__
	const char *name;
#else
	char *name;
#endif
	/* has_arg can't be an enum because some compilers complain about
	   type mismatches in all the code that assumes it is an int.  */
	/* has_arg �� ������ �� ���� �ֳ��ϸ� �Ϻ� �����Ϸ��� ��� �ڵ忡�� ���ĺ���ġ�� ����
	   int ��� �����Ͽ� �þ���´�. */
	int has_arg;
	int *flag;
	int val;
};

/* Names for the values of the `has_arg' field of `struct option'.  */
/* 'option ����'�� 'has_arg' �׸��� ���� ���� �̸�. */
#define	no_argument		0
#define required_argument	1
#define optional_argument	2

#if defined (__STDC__) && __STDC__
#ifdef __GNU_LIBRARY__
/* Many other libraries have conflicting prototypes for getopt, with
   differences in the consts, in stdlib.h.  To avoid compilation
   errors, only prototype getopt for the GNU C library.  */
extern int getopt ( int argc, char *const *argv, const char *shortopts );

#else /* not __GNU_LIBRARY__ */
extern int getopt ();

#endif /* __GNU_LIBRARY__ */
extern int getopt_long ( int argc
						, char *const *argv
						, const char *shortopts
						, const struct option *longopts
						, int *longind );

extern int getopt_long_only ( int argc
							, char *const *argv
							, const char *shortopts
							, const struct option *longopts
							, int *longind );

/* Internal only.  Users should not call this directly.  */
extern int _getopt_internal ( int argc
							, char *const *argv
							, const char *shortopts
							, const struct option *longopts
							, int *longind
							, int long_only );

#else /* not __STDC__ */
extern int getopt ();
extern int getopt_long ();
extern int getopt_long_only ();

extern int _getopt_internal ();

#endif /* __STDC__ */

#ifdef __cplusplus
}
#endif

#endif /* _GETOPT_H */
