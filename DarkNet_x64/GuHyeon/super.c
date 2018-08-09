
#include "darknet.h"

void train_super( char *cfgfile, char *weightfile, int clear )
{
	//char *train_images		= "/data/imagenet/imagenet1k.train.list";	//  [7/6/2018 jobs]
	//char *backup_directory	= "/home/pjreddie/backup/";					//  [7/6/2018 jobs]
	char *train_images		= "/GiGyeHakSeub/data/imagenet/imagenet1k.train.list";	//  [7/6/2018 jobs]
	char *backup_directory	= "/GiGyeHakSeub/weights/backup/";					//  [7/6/2018 jobs]

	srand( time( 0 ) );
	char *base = basecfg( cfgfile );
	printf( "%s\n", base );
	float avg_loss = -1;
	network *net = load_network( cfgfile, weightfile, clear );

	//printf( "Learning Rate: %g, Momentum: %g, Decay: %g\n"	//  [7/6/2018 jobs]
	printf( "학습율: %g, 가속도: %g, 감쇄: %g\n"	//  [7/6/2018 jobs]
		, net->learning_rate, net->momentum, net->decay );

	int imgs = net->batch*net->subdivisions;
	int i = *net->seen/imgs;
	data train, buffer;

	list *plist = get_paths( train_images );
	//int N = plist->size;
	char **paths = (char **)list_to_array( plist );

	load_args args = { 0 };
	args.w = net->w;
	args.h = net->h;
	args.scale = 4;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.d = &buffer;
	args.type = SUPER_DATA;

	pthread_t load_thread = load_data_in_thread( args );
	clock_t time;

	//while ( i*imgs < N*120)
	while ( get_current_batch( net ) < net->max_batches )
	{
		i += 1;
		time=clock();
		pthread_join( load_thread, 0 );
		train = buffer;
		load_thread = load_data_in_thread( args );

		//printf( "Loaded: %lf seconds\n", sec( clock()-time ) );	//  [7/6/2018 jobs]
		printf( "탑재: %lf 초\n", sec( clock()-time ) );	//  [7/6/2018 jobs]

		time=clock();
		float loss = train_network( net, train );
		if ( avg_loss < 0 ) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		//printf( "%d: %f, %f avg, %f rate, %lf seconds, %d images\n"	//  [7/6/2018 jobs]
		printf( "%d=> 오차: %f, 평균오차: %f, 학습율: %f, %lf 초, %d 이미지\n"	//  [7/6/2018 jobs]
				, i
				, loss
				, avg_loss
				, get_current_rate( net )
				, sec( clock()-time )
				, i*imgs );

		if ( i%1000==0 )
		{
			char buff[256];
			//sprintf( buff, "%s/%s_%d.weights", backup_directory, base, i );
			sprintf_s( buff, 256, "%s/%s_%d.weights", backup_directory, base, i );
			save_weights( net, buff );
		}

		if ( i%100==0 )
		{
			char buff[256];
			//sprintf( buff, "%s/%s.backup", backup_directory, base );
			sprintf_s( buff, 256, "%s/%s.backup", backup_directory, base );
			save_weights( net, buff );
		}
		free_data( train );
	}

	char buff[256];
	//sprintf( buff, "%s/%s_final.weights", backup_directory, base );
	sprintf_s( buff, 256, "%s/%s_final.weights", backup_directory, base );
	save_weights( net, buff );
}

void test_super( char *cfgfile, char *weightfile, char *filename )
{
	network *net = load_network( cfgfile, weightfile, 0 );
	set_batch_network( net, 1 );
	srand( 2222222 );

	clock_t time;
	char buff[256];
	char *input = buff;

	while ( 1 )
	{
		if ( filename )
		{
			//strncpy( input, filename, 256 );
			strncpy_s( input, 256, filename, 256 );
		}
		else
		{
			//printf( "Enter Image Path: " );	//  [7/6/2018 jobs]
			printf( "이미지 경로 입력: " );	//  [7/6/2018 jobs]
			fflush( stdout );
			input = fgets( input, 256, stdin );

			if ( !input ) return;

			//strtok( input, "\n" );
			char *NaMeoJi;
			strtok_s( input, "\n", &NaMeoJi );
		}

		image im = load_image_color( input, 0, 0 );
		resize_network( net, im.w, im.h );
		printf( "%d %d\n", im.w, im.h );

		float *X = im.data;
		time=clock();
		network_predict( net, X );
		image out = get_network_image( net );
		//printf( "%s: Predicted in %f seconds.\n", input, sec( clock()-time ) );	//  [7/6/2018 jobs]
		printf( "%s: 예상에 %f 초.\n", input, sec( clock()-time ) );	//  [7/6/2018 jobs]
		save_image( out, "out" );
		show_image( out, "out" );

		free_image( im );
		if ( filename ) break;
	}
}


void run_super( int argc, char **argv )
{
	if ( argc < 4 )
	{
		fprintf( stderr
			//, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n"	//  [7/6/2018 jobs]
			, "사용법: %s %s [train/test/valid] [구성파일(.cfg)] [가중값파일(.weights) (선택사항)]\n"	//  [7/6/2018 jobs]
			, argv[0], argv[1] );
		return;
	}

	char *cfg = argv[3];
	char *weights = (argc > 4) ? argv[4] : 0;
	char *filename = (argc > 5) ? argv[5] : 0;
	int clear = find_arg( argc, argv, "-clear" );

	if		( 0==strcmp( argv[2], "train" ) )	train_super( cfg, weights, clear );
	else if ( 0==strcmp( argv[2], "test" ) )	test_super( cfg, weights, filename );
	/*
	else if(0==strcmp(argv[2], "valid")) validate_super(cfg, weights);
	*/
}
