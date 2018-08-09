:kr:다크넷: C로 작성한 신경망 공개원본

출처:
- https://github.com/pjreddie/darknet
- https://pjreddie.com/darknet

참조: https://github.com/zeuseyera/darknet-kr


## 1. 윈도우판 다크넷(darknet_x64)

 출처(https://github.com/pjreddie/darknet)의 파일이름과 일부 함수이름 등을 변경하였음...

### 1-1) 관련 환경

  * 비주얼스튜디오
  * CUDA
  * OPENCV

### 1-2) 참조

  * x64 환경이지만 x86, Linux 환경에서도 가능  
  * 일부 설명이 실제 기능을 잘못 설명했을 수 있음  
  * 구현된 원본은 현재설정으로 동작이 안될수 있음  


## 2. 비주얼스튜디오 설정항목

  * 한글표시문제 발생시 속셩변경  
    * 프로젝트 속성 => 구성 속성 => 일반 => 프로젝트 기본값 => 문자 집합  
      => "유니코드 문자 집합 사용" => "**멀티바이트 문자 집합 사용**" 으로 설정  

### 2-1) CUDA C/C++ 파일을 빌드하기위한 설정

```
(1) "솔루션 탐색기"에서 "프로젝트 이름"을 마우스 오른쪽버튼을 누른다.  
(2) 팝업메뉴의 "빌드 종속성(B)" => "사용자 지정 빌드(B)"를 선택한다.  
(3) "사용 가능한 빌드 사용자 지정 파일(A)" 목록창이 열린다.  
(4) "CUDA x.x(.targets, .props)" 를 선택표시를 하고 확인버튼을 누른다.  

(5) 모든 "xxx.cu" 파일들의 속성을 변경한다  
 	(5-1) "xxx.cu" 파일에서 마우스 오른쪽버튼을 누른다.  
 	(5-2) 속성을 선택한다("파일 속성 페이지" 새창이 열린다).  
 	(5-3) 구성속성 => 일반 => 항목 형식  
     "빌드에 참여 안 함" => "**CUDA C/C++**" 로 변경한다.  

(6) 프로젝트 => "프로젝트 이름" 속성을 선택한다("프로젝트 속성 페이지" 새창이 열린다).  
 	(6-1) CUDA C/C++ => Common => Target Machine Platform 을 변경한다(선택사항).  
 		"32-bit(--machine 32)" 을  
 			=> "**64-bit(--machine 64)**" 로  
 	(6-2) CUDA C/C++ => Device => Code Generation 을 변경한다(선택사항).  
 		"compute_20,sm_20" 을  
 			=> "**compute_30,sm_30; compute_52,sm_52**" 로  
```

### 2-2) 프로젝트 속성을 CUDA, OPENCV 관련 선택사항을 설정한다

  - 디버그, 배포에 따라 각각의 환경을 설정한다

(1) C/C++ => 일반 => 추가 포함 디렉토리  

```
		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include  
		G:\Work_GaeBal\DoGu\opencv\build\include  
		G:\Work_GaeBal\DarkNet_x64\DarkNet_x64\3rdparty\include  
		$(CudaToolkitIncludeDir)  
		$(cudnn)\include  
```

(2) C/C++ => 전처리기 => 전처리기 정의

```
		HAVE_STRUCT_TIMESPEC	//error msb3721  
		OPENCV  
		_LIB  
		WIN32  
		GPU  
```		

(3) C/C++ => 전처리기 => 전처리기 정의 해제

```
		CUDNN
```

(4) C/C++ => 미리 컴파일된 헤더 => 미리 컴파일된 헤더

```
		"사용(/Yu)" => "**미리 컴파일된 헤더 사용 안 함**" 로 설정
```

(5) 링커 => 일반 => 추가 라이브러리 디렉토리

```
		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64
		G:\Work_GaeBal\DoGu\opencv\build\x64\vc14\lib
		G:\Work_GaeBal\DarkNet_x64\DarkNet_x64\3rdparty\lib\x64
		$(CUDA_PATH)lib\$(PlatformName)
		$(cudnn)\lib\x64
```

(6) 링커 => 입력 => 추가 종속성(디버그)

```
		ws2_32.lib				//error LNK2001: select
		pthreadVC2.lib
		opencv_highgui2413d.lib
		opencv_core2413d.lib
		opencv_imgproc2413d.lib
		cublas.lib
		curand.lib
		cudart.lib
```

(7) 링커 => 입력 => 추가 종속성(배포)

```
		ws2_32.lib				//error LNK2001: select
		pthreadVC2.lib
		opencv_highgui2413.lib
		opencv_core2413.lib
		opencv_imgproc2413.lib
		cublas.lib
		curand.lib
		cudart.lib
```

### 2-3) 실행에 필요한 CUDA, OPENCV, 쓰레드 관련 dll 파일

(1)	디버그 환경(64bit)

```
    opencv_core2413d.dll
    opencv_ffmpeg2413_64.dll
    opencv_highgui2413d.dll
    opencv_imgproc2413d.dll
    pthreadVC2.dll
```  

(2) 배포 환경(64bit)

```  
    opencv_core2413.dll
    opencv_ffmpeg2413_64.dll
    opencv_highgui2413.dll
    opencv_imgproc2413.dll
    pthreadVC2.dll
```


---
