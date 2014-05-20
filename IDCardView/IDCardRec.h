#ifndef _IDCARDREC_H
#define _IDCARDREC_H

#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <exception>
#include <vector>
#include <stdio.h>
#include <opencv/highgui.h>
#include <opencv/cv.h>


#include "IntImage.h"
#include "SimpleClassifier.h"
#include "AdaBoostClassifier.h"
#include "CascadeClassifier.h"
#include "Global.h"

#define showSteps 1
#define saveFlag 1

using namespace std;

//typedef unsigned char uchar;

int charWidth = 36;
int charHeight = 54;

const int ROIHEIGHT = 46;

struct  recCharAndP
{
	float recP;
	char recChar;
};

struct imageIpl 
{
	IplImage * roiImage;
	int positionX;
};

struct resultPos 
{
	recCharAndP recCharP;
	int recPosition;
};

struct resultFinal 
{
	string recString;
	float recPFinal;
};

void init();
void processingOne(IplImage * src);
void processingOneT(IplImage *src);
void processingOneP(IplImage *src);


//LARGE_INTEGER m_liPerfFreq;
//LARGE_INTEGER m_liPerfStart;
//LARGE_INTEGER liPerfNow;
//double dfTim;
//void getStartTime()
//{
//	QueryPerformanceFrequency(&m_liPerfFreq);
//	QueryPerformanceCounter(&m_liPerfStart);
//}
//
//void getEndTime()
//{
//	QueryPerformanceCounter(&liPerfNow);
//	dfTim=( ((liPerfNow.QuadPart - m_liPerfStart.QuadPart) * 1000.0f)/m_liPerfFreq.QuadPart);
//}

#endif
