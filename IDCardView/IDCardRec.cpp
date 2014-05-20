#include "IDCardRec.h"
#include <math.h>
using namespace cv;
#define random(x) (rand()%x)
char* trainSetPosPath = (char *)malloc(200*sizeof(char));
void readConfig(char* configFile, char* trainSetPosPath){
//	fstream f;
//	char cstring[1000];
//	int readS=0;
//	f.open(configFile, fstream::in);
//	char param1[200]; strcpy(param1,"");
//	char param2[200]; strcpy(param2,"");
//	char param3[200]; strcpy(param3,"");
//
//	f.getline(cstring, sizeof(cstring));
//	readS=sscanf (cstring, "%s %s %s", param1,param2, param3);
//	strcpy(trainSetPosPath,param3);
}


vector<string> imgNames;
int labelTemp = 0;

int bbOverlap(const Rect& box1,const Rect& box2)
{
    if (box1.x > box2.x+box2.width) { return 0.0; }
    if (box1.y > box2.y+box2.height) { return 0.0; }
    if (box1.x+box1.width < box2.x) { return 0.0; }
    if (box1.y+box1.height < box2.y) { return 0.0; }
    int colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
    int rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);
    int intersection = colInt * rowInt;
    return intersection;
}

void bubbleSort(Rect arr[],int len)
{
    int i,j;
    Rect temp;
    for(i=0;i<len;i++)
    {
        for(j=0;j<len-i-1;j++)
        {
            if(arr[j].x>arr[j+1].x)
            {
                temp=arr[j];
                arr[j]=arr[j+1];
                arr[j+1]=temp;
            }
        }
    }
}
pair<int, int> getMaxAndMin(int a[],int n)
{
    int i,min=INT16_MAX,max=0;
    for(i=0;i<n;i++){
        if(a[i]>max){
            max=a[i];
        }
        if(a[i]<min)
        {
            min=a[i];
        }
    }
    return make_pair(max, min);
}
pair<int, int> getTop2Position(int a[],int n)
{
    int i,first=-1,second=-1,max=0;
    for(i=0;i<n;i++){
        if(a[i]>max){
            max=a[i];
            second = first;
            first = i;
        }
    }
    return make_pair(first, second);
}
int searchMax(int a[], int n)
{
	int i,maxTag,max=0;
    for(i=0;i<n-5;i++)
    {
        if(a[i]+a[i+1]+a[i+2]+a[i+3]+a[i+4]>max)
        {
            max=a[i]+a[i+1]+a[i+2]+a[i+3]+a[i+4];
            maxTag=i;
        }
    }
    return maxTag;
}
vector<Rect> modifyRect(Mat mat,vector<Rect> srcList)
{
    vector<Rect> result;
    int allWidth[srcList.size()];
    for(int i=0;i<srcList.size();i++)
    {
        allWidth[i] = srcList.at(i).width;
    }
    pair<int, int> maxAndMin = getMaxAndMin(allWidth,(int)srcList.size());
    int max = maxAndMin.first;
    int min = maxAndMin.second;
    const int WIDTH = 8;
    const int count = abs(max-min)/WIDTH+1;
    int allDirect[count];
    for(int i=0;i<count;i++)
    {
       allDirect[i] = 0;
    }
    for(int i=0;i<srcList.size();i++)
    {
        int position = (srcList.at(i).width-min)/WIDTH;
        allDirect[position] = allDirect[position]+1;
    }
    int rightWidth = 30;
    pair<int, int> top2 = getTop2Position(allDirect, count);
    if(allDirect[top2.first]>srcList.size()/2.0f)
    {
        rightWidth = min+WIDTH*top2.first+WIDTH/2;
    }
    else
    {
        if(allDirect[top2.first]+allDirect[top2.second]>srcList.size()/3.0f*2&&abs(top2.first-top2.second)==1)
        {
            rightWidth = MAX(28,((min+WIDTH*top2.first+WIDTH/2)+(min+WIDTH*top2.second+WIDTH/2))/2);
        }
        else
        {
            rightWidth = MAX(28,min+WIDTH*top2.first+abs(top2.first-top2.second)*WIDTH/2);
        }
    }
    
    for (int i=0;i<srcList.size();i++) {
        float tip = srcList.at(i).width/(rightWidth*1.0f);
        if(tip<1.0f)
        {
            Rect rect(srcList.at(i).x-(rightWidth-srcList.at(i).width)/2.0f,0,rightWidth,srcList.at(i).height);
            result.push_back(rect);
        }
        else
        {
            if((int)(tip*100)-((int)tip)*100>40)
            {
                Rect rect(srcList.at(i).x-(rightWidth*((int)tip+1)-srcList.at(i).width)/2.0f,0,rightWidth*((int)tip+1),srcList.at(i).height);
                for (int j=0; j<(int)tip+1; j++) {
                    Rect miniRect(rect.x+j*rightWidth,0,rightWidth,rect.height);
                    result.push_back(miniRect);
                }
            }
            else
            {
                for (int j=0; j<(int)tip; j++) {
                    Rect rect(srcList.at(i).x+j*rightWidth,0,rightWidth,srcList.at(i).height);
                    result.push_back(rect);
                }
            }
        }
    }
    Mat drawing = Mat::zeros( mat.size(), CV_8UC3 );
    for( int i = 0; i < result.size(); i++ ){
      rectangle(drawing,result.at(i),Scalar(random(255),random(255),random(255)),1);
    }
    cvShowImage("last", new IplImage(drawing));
    return result;
}
vector<Rect> findRect(Mat imgMat,char *  str)
{
    vector<Rect> noDetectResult;
    vector<Rect> result;
    char name[256]={0};
    sprintf(name, "findRect_%s",str);
    Mat drawing = Mat::zeros( imgMat.size(), CV_8UC3 );
    vector<vector<Point> > roi_contours;
    vector<Vec4i> roi_hierarchy;
    findContours(imgMat, roi_contours, roi_hierarchy, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );
    vector<Rect> roi_boundRect( roi_contours.size() );
    for( int i = 0; i < roi_contours.size(); i++ ){
        Rect r0= boundingRect(Mat(roi_contours[i]));//boundingRect获取这个外接矩形
        if((r0.width<imgMat.cols)&&((r0.width>=imgMat.cols/40.0f&&r0.height>=imgMat.rows/2.0f)||(r0.width>=imgMat.cols/80.0f&&r0.y<=imgMat.rows*0.45f&&r0.height>=imgMat.rows*0.65f)))
        {
            r0.y=0;
            r0.height=imgMat.rows;
            rectangle(drawing,r0,Scalar(random(255),random(255),random(255)),1);
            noDetectResult.push_back(r0);
        }
    }
    const int size = (int)noDetectResult.size();
    Rect* arr = (Rect*) new Rect[size];
    for(int i=0;i<noDetectResult.size();i++)
    {
        arr[i]=noDetectResult.at(i);
    }
    bubbleSort(arr,size);
    for (int i=0;i<size-1;i++)
    {
        float overlap = bbOverlap(arr[i], arr[i+1]);
        if(overlap>0)
        {
            float areaI = arr[i].width*arr[i].height;
            float areaJ = arr[i+1].width*arr[i+1].height;
            if(areaI>=areaJ)
            {
                if(overlap/areaJ>=0.8&&overlap/areaJ>=0.5)
                {
                    Rect mergeRect(arr[i].x,0,max(arr[i+1].x+arr[i+1].width,arr[i].x+arr[i].width)-arr[i].x,imgMat.rows);
                    result.push_back(mergeRect);
                    i++;
                }
                else
                {
                    Rect reRect(arr[i].x,0,(arr[i+1].x-arr[i].x),imgMat.rows);
                    arr[i] = reRect;
                    result.push_back(arr[i]);
                }
            }
            else
            {
                if(overlap/areaI>=0.8&&overlap/areaJ>=0.5)
                {
                    Rect mergeRect(arr[i].x,0,max(arr[i+1].x+arr[i+1].width,arr[i].x+arr[i].width)-arr[i].x,imgMat.rows);
                    result.push_back(mergeRect);
                    i++;
                }
                else
                {
                    Rect reRect(arr[i].x+arr[i].width,0,arr[i+1].width-(arr[i].x+arr[i].width-arr[i+1].x),imgMat.rows);
                    arr[i+1] = reRect;
                    result.push_back(arr[i]);
                }
            }
        }
        else
        {
            result.push_back(arr[i]);
        }
        if(i==size-2)
        {
            result.push_back(arr[size-1]);
        }
    }
    for( int i = 0; i < result.size(); i++ ){
       rectangle(drawing,result.at(i),Scalar(255,255,255),1);
    }
    cvShowImage(name, new IplImage(drawing));
//    cv::imshow(name,drawing);
    return noDetectResult;
}
IplImage * toSinglePic(Mat contenMat,int index){
    
    IplImage* img= new IplImage(contenMat);//加载图像，图像放在Debug文件夹里，这里是相对路径
    int i,j;
    CvMat *samples=cvCreateMat((img->width)*(img->height),1,CV_32FC3);//创建样本矩阵，CV_32FC3代表32位浮点3通道（彩色图像）
    CvMat *clusters=cvCreateMat((img->width)*(img->height),1,CV_32SC1);//创建类别标记矩阵，CV_32SF1代表32位整型1通道
    
    int k=0;
    for (i=0;i<img->width;i++)
    {
        for (j=0;j<img->height;j++)
        {
            CvScalar s;
            //获取图像各个像素点的三通道值（RGB）
            s.val[0]=(float)cvGet2D(img,j,i).val[0];
            s.val[1]=(float)cvGet2D(img,j,i).val[1];
            s.val[2]=(float)cvGet2D(img,j,i).val[2];
            cvSet2D(samples,k++,0,s);//将像素点三通道的值按顺序排入样本矩阵
        }
    }
    
    int nCuster=2;//聚类类别数，自己修改。
    cvKMeans2(samples,nCuster,clusters,cvTermCriteria(CV_TERMCRIT_ITER,50,1.0));//开始聚类，迭代100次，终止误差1.0
    
    
    IplImage *bin=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);//创建用于显示的图像，二值图像
    k=0;
    int val=0;
    float step=255/(nCuster-1);
    
    for (i=0;i<img->width;i++)
    {
        for (j=0;j<img->height;j++)
        {
            val=(int)clusters->data.i[k++];
            CvScalar s;
            s.val[0]=255-val*step;//这个是将不同类别取不同的像素值，
            cvSet2D(bin,j,i,s);        //将每个像素点赋值
        }
    }
    char str1[60] = {0};
    sprintf(str1,"toSinglePic_%d",index);
    cvShowImage(str1, bin); //显示图像
    return bin;
    
}
pair<float, float> findYPosition(Mat source){
    Mat src;
    cvtColor( source, src, CV_BGR2GRAY);
    IplImage *ipl_img = new IplImage(src);
    IplImage *uImage = cvCreateImage(cvSize(ipl_img->width, ipl_img->height), IPL_DEPTH_8U, 1);
    cvResize(ipl_img, uImage,CV_INTER_LINEAR);
    IplImage *hHistImag = cvCreateImage(cvSize(ipl_img->width, ipl_img->height), IPL_DEPTH_8U, 1);
    cvZero(hHistImag);
    int i,j;
    int HIS_COUNT = 50;
    float HIS_HEIGTH = ipl_img->height/50.0f;
    int hSlot[HIS_COUNT];
    memset(hSlot, 0, sizeof(int)*HIS_COUNT);
    for(i=0;i<uImage->height;i++){
        for(j=0;j<uImage->width;j++){
            int tmp=cvGet2D(uImage,i,j).val[0];
            if(tmp>0){
                hSlot[(int)(i/HIS_HEIGTH)]+=1;
            }
        }
    }
    for (i=0; i<HIS_COUNT; i++) {
        cvRectangle(hHistImag, cvPoint(ipl_img->width-hSlot[i]/HIS_HEIGTH,i*HIS_HEIGTH), cvPoint(ipl_img->width,(i+1)*HIS_HEIGTH), cvScalar(255,0,0,0));
    }
    int position = searchMax(hSlot,HIS_COUNT);
    float starPosition = (position)*HIS_HEIGTH;
    float endPosition = (position+5)*HIS_HEIGTH;
    if(position>0&&position<HIS_COUNT-5){
        if(hSlot[position-1]>hSlot[position+5]){
            if(hSlot[position-1]>(hSlot[position]+hSlot[position+1]+hSlot[position+2]+hSlot[position+3]+hSlot[position+4])/5*0.7){
                starPosition = (position-1)*HIS_HEIGTH;
            }
        }else{
            if(hSlot[position+5]>(hSlot[position]+hSlot[position+1]+hSlot[position+2]+hSlot[position+3]+hSlot[position+4])/5*0.7){
                endPosition = (position+6)*HIS_HEIGTH;
            }
        }
    }
    return make_pair(starPosition, endPosition);
    
}

Rect findPosition(Mat src )
{
    Mat erosion_dst;
    Mat element = getStructuringElement(MORPH_RECT, Size( 1, 1 ),Point( -1, -1 ) );
    erode( src, erosion_dst, element);
    dilate( erosion_dst, erosion_dst, element );
    erode( erosion_dst, erosion_dst, element);
    dilate( erosion_dst, erosion_dst, element );
    erode( erosion_dst, erosion_dst, element);
    dilate( erosion_dst, erosion_dst, element );
    cvtColor(erosion_dst,erosion_dst,CV_BGR2GRAY);
    
    //    threshold(erosion_dst, erosion_dst, 48, 255, CV_THRESH_BINARY_INV);
    //    GaussianBlur( ç, erosion_dst, Size(2*erosion_size + 1, 2*erosion_size + 1 ), 0, 0, BORDER_DEFAULT );
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    /// 求 X方向梯度
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( erosion_dst, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    
    /// 求Y方向梯度
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( erosion_dst, grad_y, CV_16S, 0, 1, 3, 1, 0.4, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    /// 合并梯度(近似)
    addWeighted( abs_grad_x, 0.8, abs_grad_y, 0.2, 0, erosion_dst );
    
    threshold(erosion_dst, erosion_dst, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
    vector<vector<Point> > squares;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( erosion_dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //
    //    /// 多边形逼近轮廓 + 获取矩形和圆形边界框
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    
    for( int i = 0; i < contours.size(); i++ ){
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }
    Mat drawing = Mat::zeros( erosion_dst.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        vector<Point> points = contours_poly.at(i);
        const int count = (int)points.size();
        Point *pp = (Point*)malloc(count*sizeof(Point));
        for (int j = 0; j < count; j++) {
            pp[j] = points.at(j);
        }
        const Point* ppts[1] = {pp};
        Rect rect = boundingRect(Mat(points));
        
        if(rect.y<erosion_dst.rows*0.7&&rect.y>erosion_dst.rows*0.45&&fabs(contourArea(Mat(points))) < erosion_dst.rows*erosion_dst.cols/10){
            fillPoly(drawing, ppts, &count, 1, Scalar( 255, 255, 255 ));
        }
        free(pp);
    }
    pair<float, float> startEnd = findYPosition(drawing);
    Rect rect(0,startEnd.first,src.cols,startEnd.second-startEnd.first);
    return rect;
}


void initTrainImage(){
	readConfig(NULL, trainSetPosPath);

	string folderPath = trainSetPosPath;
//	dfsFolder(folderPath);

}

void processingTotal(){
	initTrainImage();
    IplImage * src;
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/card1.jpg");
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/card3.jpg");
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/card10.png");
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/card11.jpg");
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/card12.jpg");
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/card13.png");
    src = cvLoadImage("/Users/yrguo/Desktop/pic/card14.png");
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/card15.png");
//    src = cvLoadImage("/Users/yrguo/Desktop/pic/test.png");
    processingOne(src);
    
//	long imgNum = imgNames.size();
//	for(int iNum=0;iNum<imgNum;iNum++){
//
//		cout<<endl<<iNum<<endl;
//		cout<<imgNames[iNum].c_str()<<endl;
//		IplImage * src=cvLoadImage(imgNames[iNum].c_str(),1);  
//		if(!src) continue;
//		if(1){
//			cvNamedWindow("image",1);
//			cvShowImage("image",src);
//		}
//
//		processingOne(src);
//		
//	}
}

void init(){
	InitTestGlobalData();
}

static void OpenClose(IplImage* src,IplImage* dst,int pos)
{
	int an =1;
	IplConvKernel* element = 0;
	int element_shape = CV_SHAPE_RECT;
	element = cvCreateStructuringElementEx( an*2+1, an*2+1, an, an, element_shape, 0 );
	if( pos < 0 )
	{
		cvErode(src,dst,element,1);
		cvDilate(dst,dst,element,1);
	}
	else
	{
		cvDilate(src,dst,element,1);
		cvErode(dst,dst,element,1);
	}
	cvReleaseStructuringElement(&element);

}
void InsertSort(int a[],int count)
{
	int i,j,temp;
	for(i=1;i<count;i++)   
	{
		temp=a[i];
		j=i-1;
		while(a[j]>temp && j>=0)
		{
			a[j+1]=a[j];
			j--;
		}
		if(j!=(i-1))     
			a[j+1]=temp;
	}
}


void processingOne(IplImage * dst){
	cvNamedWindow("dst",1);
	cvShowImage("dst",dst);

	IplImage* dst1 = cvCreateImage(cvGetSize(dst),8,dst->nChannels);
	cvCopy(dst,dst1);

	processingOneT(dst1);
//	processingOneP(dst);

	cvDestroyAllWindows();
//	cvReleaseImage(&dst);
	cvReleaseImage(&dst1);




}

void processingOneT(IplImage *src){
	//processing-------------
	int _width = (int)((float)(540.0/src->height)*src->width);;
	int _height = 540;

	IplImage *srcResize = cvCreateImage(cvSize(_width,_height),8,src->nChannels);
	cvResize(src,srcResize,CV_INTER_LINEAR);
    
//    IplImage * dst = cvLoadImage("/Users/yrguo/Desktop/pic/roi_img.jpg");
//    IplImage *dstY = cvCreateImage(cvSize(dst->width,dst->height),8,1);
//	cvCvtColor(dst,dstY,CV_BGR2GRAY);
	
    CvRect posDst;
    Rect rect = findPosition(Mat(srcResize));
    posDst.x = rect.x;
    posDst.y = rect.y;
    posDst.width = rect.width;
    posDst.height = rect.height;
    
	IplImage *dst = cvCreateImage(cvSize(posDst.width,posDst.height),8,src->nChannels);
    cvSetImageROI(srcResize,posDst);
	cvCopy(srcResize,dst);
    cvResetImageROI(srcResize);
	if(showSteps){
		cvNamedWindow("dst",1);
		cvShowImage("dst",dst);
	}

//    CvRect roi;
//	roi.x = 55;
//	roi.y = 300;
//	roi.width = 769;
//	if( (roi.width + roi.x)>srcResize->width )
//		roi.width = srcResize->width - roi.x;
//	roi.height = 80;
//	IplImage *dst = cvCreateImage(cvSize(roi.width,roi.height),8,src->nChannels);
//	cvSetImageROI(srcResize,roi);
//	cvCopy(srcResize,dst);
//	cvResetImageROI(srcResize);
//	if(showSteps){
//		cvNamedWindow("dst",1);
//		cvShowImage("dst",dst);
//	}
    
	IplImage* dstTemp = cvCreateImage(cvSize(dst->width,dst->height),8,src->nChannels);
	cvCopy(dst,dstTemp);
	cvSmooth(dstTemp,dstTemp,CV_GAUSSIAN );

	
	IplImage * pImagePlanes[3]={NULL,NULL,NULL};
	IplImage * pImage16uColorSobel=NULL;
	IplImage * pImage8uColorSobelShow_x=NULL;
	IplImage * pImage8uColorSobelShow_y=NULL;
	int i;
	for (i=0;i<3;i++){
		pImagePlanes[i]=cvCreateImage(cvGetSize(dst),IPL_DEPTH_8U,1);
	}
	pImage16uColorSobel=cvCreateImage(cvGetSize(dst),IPL_DEPTH_16S,1);
	pImage8uColorSobelShow_x=cvCreateImage(cvGetSize(dst),IPL_DEPTH_8U,3);
	
    
	cvSplit(dstTemp,pImagePlanes[0],pImagePlanes[1],pImagePlanes[2],NULL);
    
    
	for (i=0;i<3;i++){
		cvSobel(pImagePlanes[i],pImage16uColorSobel,0,1,3 );
		cvConvertScaleAbs(pImage16uColorSobel,pImagePlanes[i],1,0);
	}
    cvShowImage("Rx",pImagePlanes[0]);
    
    cvShowImage("Gx",pImagePlanes[1]);
    
    cvShowImage("Bx",pImagePlanes[2]);
    
	cvMerge(pImagePlanes[0],pImagePlanes[1],pImagePlanes[2],NULL,pImage8uColorSobelShow_x);
	
	pImage8uColorSobelShow_y=cvCreateImage(cvGetSize(dst),IPL_DEPTH_8U,3);
	cvSplit(dstTemp,pImagePlanes[0],pImagePlanes[1],pImagePlanes[2],NULL);
	for (i=0;i<3;i++){
		cvSobel(pImagePlanes[i],pImage16uColorSobel,1,0,3 );
		cvConvertScaleAbs(pImage16uColorSobel,pImagePlanes[i],1,0);
	}
	cvMerge(pImagePlanes[0],pImagePlanes[1],pImagePlanes[2],NULL,pImage8uColorSobelShow_y);

    cvShowImage("Ry",pImagePlanes[0]);
    
    cvShowImage("Gy",pImagePlanes[1]);
    
    cvShowImage("By",pImagePlanes[2]);

	IplImage * imgsum =cvCreateImage(cvGetSize(dst),IPL_DEPTH_32F,3);  
	cvZero(imgsum);  
	cvAcc(pImage8uColorSobelShow_y,imgsum);  
	cvAcc(pImage8uColorSobelShow_x,imgsum);  
	
	IplImage * imgavg = cvCreateImage(cvGetSize(dst),IPL_DEPTH_8U,3);  
	cvConvertScale(imgsum,imgavg,1.0/2.0);  
	if(showSteps){
		cvNamedWindow("imgavg",1);
		cvShowImage("imgavg",imgavg);
	}

	
	IplImage * imgBin = cvCreateImage(cvGetSize(imgavg),IPL_DEPTH_8U,1);
	cvCvtColor(imgavg,imgBin,CV_BGR2GRAY);
	cvThreshold(imgBin,imgBin,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
//    cvDilate(imgBin, imgBin);
	if(showSteps){
		cvNamedWindow("imgBin",1);
		cvShowImage("imgBin",imgBin);
	}

    
    int x,y;
    CvScalar s,t;
    
	IplImage* painty=cvCreateImage( cvGetSize(imgBin),IPL_DEPTH_8U, 1 );  		
	cvZero(painty);  
	int* h=new int[imgBin->height];  		
	memset(h,0,imgBin->height*4);  

	for(y=0;y<imgBin->height;y++)
	{  
		for(x=0;x<imgBin->width;x++)  
		{  
			s=cvGet2D(imgBin,y,x);           
			if(s.val[0]==0)  
				h[y]++;       
		}     
	} 


	for(y=0;y<imgBin->height;y++)   {  
		if((imgBin->width-h[y]) <= 80)
			h[y] = imgBin->width;
	}

	for(x=0;x<painty->height;x++)  {
		for(y=x;y<painty->height;y++)  {
			if( (h[x] == h[y])&&(h[y] == painty->width)&&(y-x <= 6) ){
				for(int i=x;i<=y;i++){
					h[i] = painty->width;
				}
			}
			if( (h[x] != painty->width)&&(h[y] == painty->width)&&(abs(y-x) <= 6)&&((x == 0)||(y == 0)) ){
				for(int i=x;i<=y;i++){
					h[i] = painty->width;
				}
			} 

		}
	}

	for(x=0;x<painty->height;x++)  {
		for(y=x;y<painty->height;y++)  {
			if( (h[x] == h[y])&&(h[y] == painty->width)&&(y-x <= 15) ){
				for(int i=x;i<=y;i++){
					h[i] = painty->width;
				}
			}
			if( (h[x] != painty->width)&&(h[y] == painty->width)&&(abs(y-x) <= 6)&&((x == 0)||(y == 0)) ){
				for(int i=x;i<=y;i++){
					h[i] = painty->width;
				}
			} 

		}
	}


	for(y=0;y<imgBin->height;y++)  
	{  
		for(x=0;x<h[y];x++)  
		{             
			t.val[0]=255;  
			cvSet2D(painty,y,x,t);            
		}         
	} 
    cvShowImage("horizen",painty);
    
	int xLeft = 0;
	for(x=0;x<painty->height-2;x++){
		if ( cvGet2D(painty,x,painty->width - 1).val[0]== 0 ){
			xLeft = x;
			break;	
		}
		if( (cvGet2D(painty,x,painty->width - 1).val[0] == 255)&&(cvGet2D(painty,x+1,painty->width - 1).val[0] == 0) ){
			xLeft = x;
			break;	
		}
	}
	
	int xRight = 0;
	for(x=painty->height-1; x>0 ;x--){	
		if ( cvGet2D(painty,x,painty->width - 1).val[0]== 0 ){
			xRight = x;
			break;	
		}
		if( (cvGet2D(painty,x,painty->width - 1).val[0]== 255)&&(cvGet2D(painty,x-1,painty->width - 1).val[0] == 0) ){
			xRight = x;
			break;	
		}
	}
	if(xRight == 0)  xRight = painty->height;

	CvRect roiDst;
//	roiDst.x = 0;
//	roiDst.y = xLeft - 3;
//	if(roiDst.y < 0) roiDst.y = xLeft;
//	roiDst.width = dst->width;
//	roiDst.height = xRight - xLeft + 6;
//	//===========================================================================================//
//	roiDst.height = charHeight;
//	if(roiDst.height >= dst->height - xLeft - 6) roiDst.height = xRight - xLeft  ;

    roiDst = CvRect(rect);
    
    
//	cout<<roiDst.x<<" "<<roiDst.y<<" "<<roiDst.width<<" "<<roiDst.height<<endl;
//    CvRect roiDst;
//    Rect rect = findPosition(Mat(srcResize));
//    roiDst.x = rect.x;
//    roiDst.y = rect.y;
//    roiDst.width = rect.width;
//    roiDst.height = rect.height;
    
    IplImage *resultBin = cvCreateImage(cvSize(roiDst.width,roiDst.height),8,src->nChannels);
//    cvSetImageROI(dst,roiDst);
	cvCopy(dst,resultBin);
//    cvResetImageROI(dst);
    
	IplImage *dstY = cvCreateImage(cvSize(roiDst.width,roiDst.height),8,1);
//    cvSetImageROI(imgBin,roiDst);
	cvCopy(imgBin,dstY);
//    cvResetImageROI(imgBin);
    
	if(showSteps){
		cvNamedWindow("dstY",1);
		cvShowImage("dstY",resultBin);
	}
//    IplImage * imgBin2 = cvCreateImage(cvGetSize(imgavg),IPL_DEPTH_8U,1);
//	cvCvtColor(imgavg,imgBin2,CV_BGR2GRAY);
//    cvThreshold(imgBin2,imgBin2,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
//    cvDilate(imgBin2, imgBin2);
//    cvShowImage("imgBin2",imgBin2);
     vector<Rect> rects = findRect(Mat(dstY), "1");
     vector<Rect> result = modifyRect(Mat(dstY),rects);
    Rect* resultArr = (Rect*) new Rect[result.size()];
    for( int i = 0; i < result.size(); i++ ){
        resultArr[i] = result.at(i);
    }
    bubbleSort(resultArr,result.size());
    for(int i = 0; i<result.size();i++)
    {
        CvRect rect = CvRect(resultArr[i]);
        IplImage *resultBin = cvCreateImage(cvSize(rect.width,rect.height),8,src->nChannels);
        cvSetImageROI(dst,rect);
        cvCopy(dst,resultBin);
        cvResetImageROI(dst);
        
        char str1[60] = {0};
        sprintf(str1,"/Users/yrguo/Desktop/pic/out/roi_img_%d.jpg",i);
        cvSaveImage(str1,resultBin);
        

    }
    
    
//    toSinglePic(Mat(imgBin2),20);
    
    
//    IplImage *dstYen= cvCreateImage(cvSize(roiDst.width,roiDst.height),8,1);
//    cvEqualizeHist(dstY, dstYen);
//    cvShowImage("dstYen",dstYen);
    
	IplImage* paintx=cvCreateImage( cvGetSize(dstY),IPL_DEPTH_8U, 1 );  		
	cvZero(paintx);  		  
	int* v=new int[dstY->width];  
	memset(v,0,dstY->width*4);  


	for(x=0;x<dstY->width;x++)  
	{  
		for(y=0;y<dstY->height;y++)  
		{  
			s=cvGet2D(dstY,y,x);           
			if(s.val[0]==0)  
				v[x]++;                   
		}         
	}  

	
	for(x=0;x<dstY->width;x++)  {  
		if((dstY->height-v[x]) <= dstY->height/10.0f)
			v[x] = dstY->height;
	}

	
	for(x=0;x<paintx->width;x++)  {
		for(y=x;y<paintx->width;y++)  {
			if( (v[x] == v[y])&&(v[y] == paintx->height)&&(y-x < 3) ){
				for(int i=x;i<=y;i++){
					v[i] = paintx->height;
				}
			}
			if( (v[x] != paintx->width)&&(v[y] == paintx->width)&&(y-x <= 6) ){
				for(int i=x;i<=y;i++){
					v[i] = paintx->height;
				}
			} 
		}
	}
	
	int xZuo[145] = {0};
	int xYou[145] = {0};
	int xD[145] = {0};

	int xZNum = 0;
	int xYNum = 0;
	for(x=0;x<paintx->width-1;x++)  {
		if( (v[x]  < paintx->height)&&(  x== 0) ){
			xZuo[0] = 0;
			xZNum ++;
		}
		if( (v[x] == paintx->height)&&(v[x+1] < paintx->height) ){
			xZuo[xZNum++] = x;
		}

		if( (v[x] < paintx->height)&&(v[x+1] == paintx->height) ){
			xYou[xYNum++] = x+1;
		}
		if( (v[x]  < paintx->height)&&(  x== paintx->width-2)&&(v[x+1]  < paintx->height) ){
			xYou[xYNum++] = x;
		}
	}


	for(x=0;x<xYNum-1;x++){
		xD[x] = xZuo[x+1]-xYou[x];
		if( xD[x] <= 25){
			for(y=xYou[x];y<=xZuo[x+1];y++)  {
				v[y] = paintx->height/4;
//                v[y] = 0;
			}
		}
	}


	int xZuo1[15] = {0};
	int xYou1[15] = {0};
	int xD1[15] = {0};
	int xStepD[15] = {0};

	int xZNum1 = 0;
	int xYNum1 = 0;
	for(x=0;x<paintx->width-1;x++)  {
		if( (v[x]  < paintx->height)&&(  x== 0) ){
			xZuo1[0] = 0;
			xZNum1 ++;
		}
		if( (v[x] == paintx->height)&&(v[x+1] < paintx->height) ){
			xZuo1[xZNum1++] = x;
		}

		if( (v[x] < paintx->height)&&(v[x+1] == paintx->height) ){
			xYou1[xYNum1++] = x+1;
		}
		if( (v[x]  < paintx->height)&&(  x== paintx->width-2)&&(v[x+1]  < paintx->height) ){
			xYou1[xYNum1++] = x;
		}
	}

	vector<imageIpl> imageChar;
	for(x=0;x<xYNum1-1;x++){
		xD1[x] = xZuo1[x+1]-xYou1[x];
	}
	int flagNum = 0;
	for(x=0;x<xYNum1;x++){
		xStepD[x] = xYou1[x]-xZuo1[x];
		if(xStepD[x]>0) flagNum ++;
	}
    
    
	for(x=0;x<dstY->width;x++)
	{
		for(y=0;y<v[x];y++)
		{
			t.val[0]=255;
			cvSet2D(paintx,y,x,t);
		}
	}

	if(1){
		cvNamedWindow("垂直积分投影",1);
		cvShowImage("垂直积分投影",paintx);
	}
	//	cvWaitKey(0);

	for(x=0;x<xYNum1;x++){
		if(( xStepD[x]>=110 )&&(xStepD[x]<175)){
			for(int ii = 0;ii<4;ii ++){
				CvRect roiChar;
				roiChar.x = xZuo1[x] + ii*charWidth - 1;
				if(roiChar.x<0) roiChar.x = 0;
				roiChar.y = roiDst.y;
				roiChar.width = charWidth+2;
				if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
				roiChar.height = roiDst.height;
				if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 

				if(roiChar.x < 0) roiChar.x = 0;
				if(roiChar.y < 0) roiChar.y = 0;
				if(roiChar.width < 0)	{roiChar.width = 1;		break;}
				if(roiChar.height < 0)	{roiChar.height = 1;		break;}

				IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
				cvSetImageROI(dst,roiChar);
				cvCopy(dst,dstChar);
				cvResetImageROI(dst);

				imageIpl roiImageChar;
				roiImageChar.roiImage = dstChar;
				roiImageChar.positionX = roiChar.x;
				imageChar.push_back(roiImageChar);
			}
		}
	
		if( (( xStepD[x]>165 )&&(xStepD[x]<230))&&((( xStepD[x+1]>115 )&&(xStepD[x+1]<175))||(( xStepD[x-1]>115 )&&(xStepD[x-1]<175))) ){
			if(xD1[x-1] > 28){
				for(int ii = 0;ii<4;ii ++){
					CvRect roiChar;
					roiChar.x = xZuo1[x] + ii*charWidth;
					roiChar.y = roiDst.y;
					roiChar.width = charWidth+4;
					if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
					roiChar.height = roiDst.height;
					if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 

					if(roiChar.x < 0) roiChar.x = 0;
					if(roiChar.y < 0) roiChar.y = 0;
					if(roiChar.width < 0)	{roiChar.width = 1;		break;}
					if(roiChar.height < 0)	{roiChar.height = 1;		break;}

					IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
					cvSetImageROI(dst,roiChar);
					cvCopy(dst,dstChar);
					cvResetImageROI(dst);

					imageIpl roiImageChar;
					roiImageChar.roiImage = dstChar;
					roiImageChar.positionX = roiChar.x;
					imageChar.push_back(roiImageChar);

				}

			}
		}

		if( (( xStepD[x]>165 )&&(xStepD[x]<230))&&((( xStepD[x+1]>115 )&&(xStepD[x+1]<175))||(( xStepD[x-1]>115 )&&(xStepD[x-1]<175))) ){
			if(xD1[x] > 28){
				for(int ii = 1;ii<=4;ii ++){
					CvRect roiChar;
					roiChar.x = xYou1[x] - ii*charWidth;
					roiChar.y = roiDst.y;
					roiChar.width = charWidth+4;
					if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
					roiChar.height = roiDst.height;
					if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 

					if(roiChar.x < 0) roiChar.x = 0;
					if(roiChar.y < 0) roiChar.y = 0;
					if(roiChar.width < 0)	{roiChar.width = 1;		break;}
					if(roiChar.height < 0)	{roiChar.height = 1;		break;}

					IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
					cvSetImageROI(dst,roiChar);
					cvCopy(dst,dstChar);
					cvResetImageROI(dst);

					imageIpl roiImageChar;
					roiImageChar.roiImage = dstChar;
					roiImageChar.positionX = roiChar.x;
					imageChar.push_back(roiImageChar);

				}

			}
		}

	
		if(( xStepD[x]>300 )&&(xStepD[x]<370)){		
			if( (x==0)&&(xD1[x]>30) ){
				for(int ii = 0;ii<8;ii ++){
					CvRect roiChar;
					if(ii>=4)
						roiChar.x = xYou1[x] - (8-ii)*charWidth;
					else 
						roiChar.x = xYou1[x] - (8-ii)*charWidth -36;
					roiChar.y = roiDst.y;
					roiChar.width = charWidth+2;
					if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
					roiChar.height = roiDst.height;;
					if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 

					if(roiChar.x < 0) roiChar.x = 0;
					if(roiChar.y < 0) roiChar.y = 0;
					if(roiChar.width < 0)	{roiChar.width = 1;		break;}
					if(roiChar.height < 0)	{roiChar.height = 1;		break;}

					IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
					cvSetImageROI(dst,roiChar);
					cvCopy(dst,dstChar);
					cvResetImageROI(dst);

					imageIpl roiImageChar;
					roiImageChar.roiImage = dstChar;
					roiImageChar.positionX = roiChar.x;
					imageChar.push_back(roiImageChar);

				}

			}else{
				for(int ii = 0;ii<8;ii ++){
					CvRect roiChar;
					if(ii<4)
						roiChar.x = xZuo1[x] + ii*charWidth;
					else 
						roiChar.x = xZuo1[x] + ii*(charWidth) + 36;
					roiChar.y = roiDst.y;
					roiChar.width = charWidth+2;
					if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
					roiChar.height = roiDst.height;;
					if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 

					if(roiChar.x < 0) roiChar.x = 0;
					if(roiChar.y < 0) roiChar.y = 0;
					if(roiChar.width < 0)	{roiChar.width = 1;		break;}
					if(roiChar.height < 0)	{roiChar.height = 1;		break;}

					IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
					cvSetImageROI(dst,roiChar);
					cvCopy(dst,dstChar);
					cvResetImageROI(dst);

					imageIpl roiImageChar;
					roiImageChar.roiImage = dstChar;
					roiImageChar.positionX = roiChar.x;
					imageChar.push_back(roiImageChar);
				}
			}

		}



		if(( xStepD[x]>480 )&&(xStepD[x]<=555)){	
			if( (( xStepD[x-1]>130 )&&(xStepD[x-1]<175))||(( xStepD[x+1]>130 )&&(xStepD[x+1]<180)) ){
				for(int ii = 0;ii<12;ii ++){
					CvRect roiChar;
					if(ii<4)
						roiChar.x = xZuo1[x] + ii*charWidth;							
					else if(ii<8)
						roiChar.x = xZuo1[x] + ii*charWidth + 36;
					else 
						roiChar.x = xZuo1[x] + ii*charWidth + 72;
					roiChar.y = roiDst.y;
					roiChar.width = charWidth+4;
					if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
					roiChar.height = roiDst.height;
					if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 

					if(roiChar.x < 0) roiChar.x = 0;
					if(roiChar.y < 0) roiChar.y = 0;
					if(roiChar.width < 0)	{roiChar.width = 1;		break;}
					if(roiChar.height < 0)	{roiChar.height = 1;		break;}

					IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
					cvSetImageROI(dst,roiChar);
					cvCopy(dst,dstChar);
					cvResetImageROI(dst);

					imageIpl roiImageChar;
					roiImageChar.roiImage = dstChar;
					roiImageChar.positionX = roiChar.x;
					imageChar.push_back(roiImageChar);
				}
			}

		}
	
		if( (( xStepD[x]>150 )&&(xStepD[x]<240))&&(( xStepD[x+1]>440 )&&(xStepD[x+1]<490)) ){
			
			for(int ii = 0;ii<6;ii ++){
				CvRect roiChar;
				roiChar.x = xZuo1[x] + ii*charWidth;										
				roiChar.y = roiDst.y;
				roiChar.width = charWidth+2;
				if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
				roiChar.height = roiDst.height;
				if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 
				if(roiChar.x < 0) roiChar.x = 0;
				if(roiChar.y < 0) roiChar.y = 0;
				if(roiChar.width < 0)	{roiChar.width = 1;		break;}
				if(roiChar.height < 0)	{roiChar.height = 1;		break;}

				IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
				cvSetImageROI(dst,roiChar);
				cvCopy(dst,dstChar);
				cvResetImageROI(dst);

				imageIpl roiImageChar;
				roiImageChar.roiImage = dstChar;
				roiImageChar.positionX = roiChar.x;
				imageChar.push_back(roiImageChar);
			}
			for(int ii = 0;ii<13;ii ++){

				CvRect roiChar;
				roiChar.x = xZuo1[x+1] + ii*(charWidth);										
				roiChar.y = roiDst.y;
				roiChar.width = charWidth+4;
				if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
				roiChar.height = roiDst.height;
				if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 
				if(roiChar.x < 0) roiChar.x = 0;
				if(roiChar.y < 0) roiChar.y = 0;
				if(roiChar.width < 0)	{roiChar.width = 1;		break;}
				if(roiChar.height < 0)	{roiChar.height = 1;		break;}

				IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
				cvSetImageROI(dst,roiChar);
				cvCopy(dst,dstChar);
				cvResetImageROI(dst);

				imageIpl roiImageChar;
				roiImageChar.roiImage = dstChar;
				roiImageChar.positionX = roiChar.x;
				imageChar.push_back(roiImageChar);
			}

		}
	
		if( (( xStepD[x]>440 )&&(xStepD[x]<490))&&(( xStepD[x+1]>150 )&&(xStepD[x+1]<240)) ){
		
			for(int ii = 0;ii<13;ii ++){
				CvRect roiChar;
				roiChar.x = xZuo1[x] + ii*charWidth;										
				roiChar.y = roiDst.y;
				roiChar.width = charWidth;
				
				if( (roiChar.width + roiChar.x) > xYou1[x] ) roiChar.width = xYou1[x] -roiChar.x -1; 
				roiChar.height = roiDst.height;
				if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 
				
				if(roiChar.x < 0) roiChar.x = 0;
				if(roiChar.y < 0) roiChar.y = 0;
				if(roiChar.width < 0)	{roiChar.width = 1;		break;}
				if(roiChar.height < 0)	{roiChar.height = 1;		break;}

				IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
				cvSetImageROI(dst,roiChar);
				cvCopy(dst,dstChar);
				cvResetImageROI(dst);

				imageIpl roiImageChar;
				roiImageChar.roiImage = dstChar;
				roiImageChar.positionX = roiChar.x;
				if(ii!=6)
					imageChar.push_back(roiImageChar);
			}
			for(int ii = 0;ii<6;ii ++){
				CvRect roiChar;
				roiChar.x = xZuo1[x+1] + ii*(charWidth+1);	
				if(roiChar.x>dst->width)	 break;								
				roiChar.y = roiDst.y;
				roiChar.width = charWidth+4;
				if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
				roiChar.height = roiDst.height;
				if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 
				
				if(roiChar.x < 0) roiChar.x = 0;
				if(roiChar.y < 0) roiChar.y = 0;
				if(roiChar.width < 0)	{roiChar.width = 1;		break;}
				if(roiChar.height < 0)	{roiChar.height = 1;		break;}

				IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
				cvSetImageROI(dst,roiChar);
				cvCopy(dst,dstChar);
				cvResetImageROI(dst);

				imageIpl roiImageChar;
				roiImageChar.roiImage = dstChar;
				roiImageChar.positionX = roiChar.x;
				imageChar.push_back(roiImageChar);

			}

		}
		
		if( ( ( (( xStepD[x]>150 )&&(xStepD[x]<240))&&((( xStepD[x+1]>150 )&&(xStepD[x+1]<240))&&(( xStepD[x+2]>150 )&&(xStepD[x+2]<240))) )||
			( (( xStepD[x-1]>150 )&&(xStepD[x-1]<240))&&((( xStepD[x ]>150 )&&(xStepD[x ]<240))&&(( xStepD[x+1]>150 )&&(xStepD[x+1]<240))) )||
			( (( xStepD[x-2]>150 )&&(xStepD[x-2]<240))&&((( xStepD[x-1]>150 )&&(xStepD[x-1]<240))&&(( xStepD[x]>150 )&&(xStepD[x]<240))) )

			)&&(flagNum == 3) ){
				
				for(int ii = 0;ii<6;ii ++){
					CvRect roiChar;
					roiChar.x = xZuo1[x] + ii*(charWidth );										
					roiChar.y = roiDst.y;
					roiChar.width = charWidth+4;
					if( (roiChar.width + roiChar.x) > dst->width ) roiChar.width = dst->width -roiChar.x -1; 
					roiChar.height = roiDst.height;
					if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 

					if(roiChar.x < 0) roiChar.x = 0;
					if(roiChar.y < 0) roiChar.y = 0;
					if(roiChar.width < 0)	{roiChar.width = 1;		break;}
					if(roiChar.height < 0)	{roiChar.height = 1;		break;}

					IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
					cvSetImageROI(dst,roiChar);
					cvCopy(dst,dstChar);
					cvResetImageROI(dst);

					imageIpl roiImageChar;
					roiImageChar.roiImage = dstChar;
					roiImageChar.positionX = roiChar.x;
					imageChar.push_back(roiImageChar);

				}
		}
		
		if( (( xStepD[x]>640 )&&(xStepD[x]<765))&&(x==0) ){
			
			for(int ii = 0;ii<19;ii ++){
				CvRect roiChar;
				roiChar.x = xZuo1[x] + ii*(charWidth);										
				roiChar.y = roiDst.y;
				roiChar.width = charWidth+2;
				roiChar.height = roiDst.height;
				
				
				if( (roiChar.width + roiChar.x) > xYou1[x] ) roiChar.width = xYou1[x] -roiChar.x -1; 	
				if( (roiChar.y + roiChar.height)>dst->height ) roiChar.height = dst->height -roiDst.y; 
				if(roiChar.x < 0) roiChar.x = 0;
				if(roiChar.y < 0) roiChar.y = 0;
				if(roiChar.width < 0)	{roiChar.width = 1;		break;}
				if(roiChar.height < 0)	{roiChar.height = 1;		break;}


				IplImage *dstChar = cvCreateImage(cvSize(roiChar.width,roiChar.height),8,3);
				cvSetImageROI(dst,roiChar);
				cvCopy(dst,dstChar);
				cvResetImageROI(dst);

				imageIpl roiImageChar;
				roiImageChar.roiImage = dstChar;
				roiImageChar.positionX = roiChar.x;
				imageChar.push_back(roiImageChar);

			}
		}

	}

	long imgNum1 = imageChar.size();
//	for(int iiNum=0;iiNum<imgNum1;iiNum++){
//		
//		if(saveFlag){
//			char name[500];
//			sprintf(name,"%s%s%d%s","/Users/yrguo/Desktop/pic/out/","_",iiNum,".bmp");
//			
//			cvSaveImage(name,imageChar[iiNum].roiImage);
//		}
//		
//	}


	

	vector<imageIpl>(imageChar).swap(imageChar);
	
	
	cvWaitKey(0);


	cvReleaseImage(&srcResize);
	cvReleaseImage(&dst);
//	cvReleaseImage(&dstTemp);
	cvReleaseImage(&imgBin);
	cvReleaseImage(&dstY);

//	cvReleaseImage(&pImagePlanes[0]);
//	cvReleaseImage(&pImagePlanes[1]);
//	cvReleaseImage(&pImagePlanes[2]);
//	cvReleaseImage(&pImage16uColorSobel);
//	cvReleaseImage(&pImage8uColorSobelShow_x);
//	cvReleaseImage(&pImage8uColorSobelShow_y);
//	cvReleaseImage(&imgsum);
//	cvReleaseImage(&imgavg);
//	cvReleaseImage(&paintx);  
//	cvReleaseImage(&painty);  

}

void processingOneP(IplImage *src){
	 
		if(!src) printf("Error!\n");

		
		int _width = (int)((float)(540.0/src->height)*src->width);;
		int _height = 540;

		IplImage *srcResize = cvCreateImage(cvSize(_width,_height),8,src->nChannels);
		cvResize(src,srcResize,CV_INTER_CUBIC);

		if(showSteps){
			cvNamedWindow("srcResize",1);
			cvShowImage("srcResize",srcResize);
		}

		
//		CvRect roi;
//		roi.x = 40;
//		roi.y = 200;
//		roi.width = srcResize->width - 20;
//		if( (roi.width + roi.x)>srcResize->width )
//			roi.width = srcResize->width - roi.x;
//		roi.height = 170;
//		IplImage *dst = cvCreateImage(cvSize(roi.width,roi.height),8,src->nChannels);
//		cvSetImageROI(srcResize,roi);
//		cvCopy(srcResize,dst);
//		cvResetImageROI(srcResize);
		
        CvRect posDst;
        Rect rect = findPosition(Mat(srcResize));
        posDst.x = rect.x;
        posDst.y = rect.y;
        posDst.width = rect.width;
        posDst.height = rect.height;
    
        IplImage *dst = cvCreateImage(cvSize(posDst.width,posDst.height),8,src->nChannels);
        cvSetImageROI(srcResize,posDst);
        cvCopy(srcResize,dst);
        cvResetImageROI(srcResize);
    
		IplImage * input_image = cvCreateImage(cvGetSize(dst),8,1);
		cvCvtColor(dst,input_image,CV_BGR2GRAY);

		IplImage* dstImage = cvCreateImage(cvGetSize(input_image),8,1);
		cvCopy(input_image,dstImage);

	
		cvNormalize(dstImage,dstImage,255,0,CV_MINMAX);
	
		IntImage image;
			const CvSize csize = cvSize(dstImage->width,dstImage->height);
			image.SetSize(csize);
			for(int i=0; i<dstImage->height; i++){
				uchar* pimg = (uchar*)(dstImage->imageData+input_image->widthStep*i);
				for(int j=0;j<dstImage->width;j++)	image.data[i][j] = pimg[j];

			}

			const char * fileName = "1";
			cascade->ApplyOriginalSize(image,fileName);
			
    cvShowImage("original", dstImage);
			CvRect ROI;
			vector<CvRect> detectROI;
			for(int num = 0;num< (int)resultROIs.size(); num++){
				ROI.x = resultROIs.at(num).x;
				ROI.y = resultROIs.at(num).y;
				ROI.width = resultROIs.at(num).width;	
				ROI.height = resultROIs.at(num).height;
				if(ROI.x < 0) ROI.x = 0;
				if(ROI.y < 0) ROI.y = 0;
				if((ROI.x + ROI.width) > dstImage->width) ROI.width = dstImage->width - ROI.x;
				if((ROI.y + ROI.height) > dstImage->height) ROI.height = dstImage->height - ROI.y;

			
				///////////////////////////////////////////////////////////////////////////////////////
				if( (ROI.width>=20)&&(ROI.height>=30) ){
					detectROI.push_back(ROI);
				}			
			}
 
			cout<<detectROI.size()<<endl;

			resultFinal result;
			if(detectROI.size() > 0){
			
				int detectROIyavg = 0;
				for(int i=0;i<detectROI.size();i++){
					detectROIyavg += detectROI[i].y;
				}
				detectROIyavg = detectROIyavg/detectROI.size();

				
				int *detectROIX;
				detectROIX = (int *)malloc(detectROI.size()*sizeof(int));
				for(int i=0;i<detectROI.size();i++){
					
					if( (detectROI[i].y >(detectROIyavg - 20))&&(detectROI[i].y < (detectROIyavg + 20)) )
						detectROIX[i] = detectROI[i].x;
				}
				InsertSort(detectROIX,(int)detectROI.size());

				vector<CvRect> detectResult;

				for(int j=0;j<detectROI.size();j++){
					for(int i=0;i<detectROI.size();i++){	
						if(detectROI[i].x == detectROIX[j]){
							cout<<detectROIX[i]<<" ";

							detectResult.push_back(detectROI[i]);

						}
					}
				}

			
				int ROIWIDTH = 30;
				for(int i=0;i<detectResult.size();i++){
			
					ROIWIDTH += detectResult[i].width;
				}
				if(detectResult.size() != 0)
					ROIWIDTH = ROIWIDTH/detectResult.size();

				
				vector<CvRect> noDetectResult;
				for(int i=0;i<detectResult.size()-1;i++){
					if(
						(detectResult[i].x + detectResult[i].width + ROIWIDTH/2) < (detectResult[i+1].x )
						){
						int j = ( detectResult[i+1].x - detectResult[i].x ) / ROIWIDTH  - 1;

					
						if( ((detectResult[i+1].x - detectResult[i].x) % ROIWIDTH ) > ROIWIDTH/2  ){
							j++;
						}
						for(int n =0 ;n<j;n++){
							CvRect Roi;
							Roi.x = detectResult[i].x  /*+ ROIWIDTH*n*/ + (ROIWIDTH )*(n+1);
						
							Roi.y = detectResult[i].y;
							Roi.width = ROIWIDTH;
							Roi.height = ROIHEIGHT;

						//	cvRectangle(dst, cvPoint(Roi.x , Roi.y ),cvPoint(Roi.x+Roi.width,Roi.y+Roi.height),CV_RGB(255,255,255), 2);	

							noDetectResult.push_back(Roi);
						}
					}

				}
				vector<CvRect> DetectAllResult;
				for(int i=0;i<detectResult.size();i++)		DetectAllResult.push_back(detectResult[i]);

			
				if(noDetectResult.size()!=0){
					for(int i=0;i<noDetectResult.size();i++)		DetectAllResult.push_back(noDetectResult[i]);
				}
				

				vector<CvRect>(detectResult).swap(detectResult);
				vector<CvRect>(noDetectResult).swap(noDetectResult);
				vector<CvRect>(detectROI).swap(detectROI);

				cout<<DetectAllResult.size()<<endl;

			
				if(DetectAllResult.size()<=23){
					int *detectROIXX;
					detectROIXX = (int *)malloc(DetectAllResult.size()*sizeof(int));
					for(int i=0;i<DetectAllResult.size();i++){
						detectROIXX[i] = DetectAllResult[i].x;
					}
					InsertSort(detectROIXX,(int)DetectAllResult.size());

					vector<CvRect> detectResultF;

					for(int j=0;j<DetectAllResult.size();j++){
						for(int i=0;i<DetectAllResult.size();i++){
						//	cout<<detectROIXX[i]<<endl;
							if(DetectAllResult[i].x == detectROIXX[j]){
								detectResultF.push_back(DetectAllResult[i]);

							}
						}
					}

					for(int i=0;i<detectResultF.size();i++){
						cout<<"==**=="<<detectResultF[i].x<<" "<<detectResultF[i].y<<" "<<detectResultF[i].width<<" "<<detectResultF[i].height<<endl;		
					}

					if(detectResultF.size()!=0){
						if(detectResultF[0].x/ROIWIDTH > 0){
							CvRect temp;
							temp.x = detectResultF[0].x -ROIWIDTH;	temp.y = detectResultF[0].y;
							temp.width = detectResultF[0].x - temp.x;    temp.height = detectResultF[0].y+detectResultF[0].height - temp.y;

							DetectAllResult.push_back(temp);
							//cvRectangle(dst, cvPoint(detectResultF[0].x -30 , detectResultF[0].y ),
							//	cvPoint(detectResultF[0].x ,detectResultF[0].y+detectResultF[0].height),CV_RGB(255,255,255), 2);	
						}
					}
					

					int tempNum = (int)detectResultF.size() - 1;
					int tempFlagF = ( dstImage->width - (detectResultF[tempNum].x+detectResultF[tempNum].width) )/ROIWIDTH ;
					if( tempFlagF > 0){
					
						for(int iR = 0;iR<(tempFlagF-2);iR++){
							CvRect temp;
							temp.x = (detectResultF[tempNum].x+detectResultF[tempNum].width) + ROIWIDTH*(iR);	  temp.y = detectResultF[tempNum].y;
							temp.width = ROIWIDTH;    temp.height = ROIHEIGHT;

							DetectAllResult.push_back(temp);
							//cvRectangle(dst, cvPoint(detectResultF[0].x -30 , detectResultF[0].y ),
							//	cvPoint(detectResultF[0].x ,detectResultF[0].y+detectResultF[0].height),CV_RGB(255,255,255), 2);	
						}		
					}
				}


				vector<resultPos> imageCharRecPos;
				for(int i=0;i<DetectAllResult.size();i++){
					if(DetectAllResult[i].x < 0) DetectAllResult[i].x = 0;
					if(DetectAllResult[i].y < 0) DetectAllResult[i].y = 0;
					if( (DetectAllResult[i].x + DetectAllResult[i].width) > dstImage->width) DetectAllResult[i].width = dstImage->width - DetectAllResult[i].x;
					if( (DetectAllResult[i].y + DetectAllResult[i].height) > dstImage->height) DetectAllResult[i].height = dstImage->height - DetectAllResult[i].y;
					
					cout<<"====="<<DetectAllResult[i].x<<" "<<DetectAllResult[i].y<<" "<<DetectAllResult[i].width<<" "<<DetectAllResult[i].height<<endl;
			
					cvSetImageROI(dst,DetectAllResult[i]);
					IplImage*  imgTempRoi = cvCreateImage(cvSize(DetectAllResult[i].width,DetectAllResult[i].height),8,3);
					cvCopy(dst,imgTempRoi);
					cvResetImageROI(dst);

			
				
					char image_name[500] ;
					sprintf(image_name, "%s%s%d%s%d%s", "/Users/yrguo/Desktop/pic/out/", "_", 1,"_", i, ".bmp");
					cvSaveImage(image_name, imgTempRoi);
					cvReleaseImage(&imgTempRoi);
				}
				
				

				
			}
			
		
			if(saveFlag){
				char image_name1[500] ;
				sprintf(image_name1, "%s%s", "/Users/yrguo/Desktop/pic/out/result", ".bmp");
				cvSaveImage(image_name1, src);
			}

			if(showSteps){
				cvNamedWindow("Source Image",1);
				cvShowImage("Source Image",dst);
			}
			
			
		
			cvWaitKey(0);
			cvDestroyAllWindows();

		
			cvReleaseImage(&srcResize);
			cvReleaseImage(&dst);
			cvReleaseImage(&input_image);
			cvReleaseImage(&dstImage);

			image.Clear();

		

}

int main(int argc, const char * argv[])
{
    
    init();
	processingTotal();
    return 0;
}