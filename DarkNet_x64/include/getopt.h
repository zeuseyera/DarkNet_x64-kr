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
/* 'getopt'에서 호출자로 통신을 위하여.
   'getopt'가 결정을 가진 선택을 찾으면, 결정 값은 여기에 반환한다.
   또한, 'odering'은 RETURN_IN_ORDER이다, 각각의 비-선택 ARGV-요소는 여기에 반환된 것이다. */
extern char *optarg;

/* Index in ARGV of the next element to be scanned.
   This is used for communication to and from the caller
   and for communication between successive calls to `getopt'.

   On entry to `getopt', zero means this is the first call; initialize.

   When `getopt' returns EOF, this is the index of the first of the
   non-option elements that the caller should itself scan.

   Otherwise, `optind' communicates from one call to the next
   how much of ARGV has been scanned so far.  */
/* ARGV 에서 훓기위한 다음 요소의 색인.
   이것은 호출자와의 통신에 사용된다 그리고 'getopt'에 대한 연속적인 호출 사이의 통신에 사용된다.
   
   'getopt' 에 들어가면, 0 은 첫번째 호출을 의미한다; 초기화 한다.
   
   'getopt' 가 EOF 을 반환할 때, 이것은 호출자가 스스로 훓어야하는 비-선택 요소의 첫번째 색인이다.
   
   그렇지 않으면, 'optind' 는 ARGV가 지금까지 얼마나 많이 훓었는지
   하나의 호출에서 다음 호출로 통신한다. */
extern int optind;

/* Callers store zero here to inhibit the error message `getopt' prints
   for unrecognized options.  */
/* 호출자는 인식할 수 없는 선택항에 대하여 오류 메시지를 방지하기 위하여 `getopt' 출력에 0을 저장한다.*/
extern int opterr;

/* Set to an option character which was unrecognized.  */
/* 인식할 수 없는 일때 선택항 문자로 설정한다.*/
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
/* 응용프로그램에서 요청한 긴 이름 선택항을 묘사한다.
   getopt_long 또는 getopt_long_only 에 LONG_OPTIONS 결정은 이름이 0 을 방지하는 요소로
   종료된 'option 구조' 의 벡터이다.
   
   'has_arg' 항목:
   no_argument			(또는 0): 선택항이 결정을 가지지 못한 경우
   required_argument	(또는 1): 선택항이 결정이 필요한 경우
   optional_argument	(또는 2): 선택항이 선택항 결정을 가진 경우
   
   'flag' 항목이 NULL 이 아닌 경우, 선택항이 발견되면 'val' 항목에 주어진 값으로 설정하고
   그 변수를 가리킨다, 하지만 선택항을 찾을 수 없는 경우는 변경되지 않는다.
   
   long-named 선택항을 사용하려면 compiled-in 상수에 'int'로 설정하고, 'optarg'에서 값을 설정하는
   것과 같이, 선택항들을 'flag' 항목을 0 으로 그리고 'val' 항목을 0 이 아닌 값으로 설정한다
   (동등한 단일-문자 선택항 문자, 하나가 있다면). 긴 선택항에 대하여 'flag' 항목이 0 인 경우,
   'getopt' 는 'val' 항목의 내용을 반환한다. */
struct option
{
#if defined (__STDC__) && __STDC__
	const char *name;
#else
	char *name;
#endif
	/* has_arg can't be an enum because some compilers complain about
	   type mismatches in all the code that assumes it is an int.  */
	/* has_arg 는 열거할 수 없다 왜냐하면 일부 컴파일러는 모든 코드에서 형식불일치에 대해
	   int 라고 가정하여 늘어놓는다. */
	int has_arg;
	int *flag;
	int val;
};

/* Names for the values of the `has_arg' field of `struct option'.  */
/* 'option 구조'의 'has_arg' 항목의 값에 대한 이름. */
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
