#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

list *read_data_cfg( char *filename )
{
	//FILE *file = fopen(filename, "r");
	FILE *file; fopen_s( &file, filename, "r" );

	if ( file == 0 ) file_error( filename );
	char *line;
	int nu = 0;
	list *options = make_list();
	while ( (line=fgetl( file )) != 0 )
	{
		++nu;
		strip( line );
		switch ( line[0] )
		{
		case '\0':
		case '#':
		case ';':
			free( line );
			break;
		default:
			if ( !read_option( line, options ) )
			{
				fprintf( stderr
					//, "Config file error line %d, could parse: %s\n"	//  [7/7/2018 jobs]
					, "구성파일 오류발생 행: %d, 분석내용: %s\n"	//  [7/7/2018 jobs]
					, nu, line );
				free( line );
			}
			break;
		}
	}
	fclose( file );
	return options;
}

metadata get_metadata( char *file )
{
	metadata m = { 0 };
	list *options = read_data_cfg( file );

	char *name_list = option_find_str( options, "names", 0 );
	if ( !name_list ) name_list = option_find_str( options, "labels", 0 );
	if ( !name_list )
	{
		fprintf( stderr
			//, "No names or labels found\n" );	//  [7/7/2018 jobs]
			, "이름이 없거나 꼬리표를 못찾음\n" );	//  [7/7/2018 jobs]
	}
	else
	{
		m.names = get_labels( name_list );
	}
	m.classes = option_find_int( options, "classes", 2 );
	free_list( options );
	return m;
}

int read_option( char *str, list *options )
{
	size_t ii;
	size_t len	= strlen( str );
	char *val	= 0;

	for ( ii=0; ii < len; ++ii )
	{
		if ( str[ii] == '=' )
		{
			str[ii]	= '\0';
			val		= str+ii+1;
			break;
		}
	}

	if ( ii == len-1 )
		return 0;

	char *key = str;
	option_insert( options, key, val );

	return 1;
}

void option_insert( list *lst, char *key, char *val )
{
	kvp *kp		= malloc( sizeof( kvp ) );
	kp->key		= key;
	kp->val		= val;
	kp->used	= 0;	// 선택사항 사용상태 미사용(0)으로 설정

	list_insert( lst, kp );
}

void option_unused( list *lst )
{
	node *nd = lst->front;

	while ( nd )
	{
		kvp *kp = (kvp *)nd->val;
		if ( !kp->used )
		{
			fprintf( stderr
				//, "Unused field: '%s = %s'\n"	//  [7/7/2018 jobs]
				, "미사용 선택사항: '%s = %s'\n"	//  [7/7/2018 jobs]
				, kp->key, kp->val );
		}

		nd = nd->next;
	}
}

char *option_find( list *lst, char *key )
{
	node *nd = lst->front;

	while ( nd )
	{
		kvp *kp = (kvp *)nd->val;

		if ( strcmp( kp->key, key ) == 0 )
		{
			kp->used = 1;	// 선택사항 사용상태 사용(1)으로 설정
			return kp->val;
		}

		nd = nd->next;
	}

	return 0;
}

char *option_find_str( list *lst, char *key, char *def )
{
	char *val = option_find( lst, key );
	if ( val ) return val;

	if ( def )
		fprintf( stderr
			//, "%s: Using default '%s'\n"	//  [7/7/2018 jobs]
			, "%s: 사용 기본값은 '%s'\n"	//  [7/7/2018 jobs]
			, key, def );
	return def;
}

int option_find_int( list *lst, char *key, int def )
{
	char *kv = option_find( lst, key );
	if ( kv ) return atoi( kv );

	fprintf( stderr
		//, "%s: Using default '%d'\n"	//  [7/7/2018 jobs]
		, "%s: 사용 기본값은 '%d'\n"	//  [7/7/2018 jobs]
		, key, def );

	return def;
}

int option_find_int_quiet( list *lst, char *key, int def )
{
	char *kv = option_find( lst, key );
	if ( kv ) return atoi( kv );

	return def;
}

float option_find_float_quiet( list *l, char *key, float def )
{
	char *v = option_find( l, key );
	if ( v ) return (float)atof( v );
	return def;
}

float option_find_float( list *l, char *key, float def )
{
	char *v = option_find( l, key );
	if ( v ) return (float)atof( v );

	fprintf( stderr
		//, "%s: Using default '%lf'\n"	//  [7/7/2018 jobs]
		, "%s: 사용 기본값은 '%lf'\n"	//  [7/7/2018 jobs]
		, key, def );
	return def;
}
