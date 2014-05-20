#include <iostream>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cvaux.hpp>
#include <opencv/cv.hpp>
#include <types_c.h>
#include "erode.h"

using namespace std;

using namespace cv;

/// 全局变量
Mat src, erosion_dst, dilation_dst,canny_dst;

int erosion_elem = 0;
int erosion_size = 0;
int erosion_time = 1;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

/** Function Headers */
void findPosition( int, void* );
void kirsch(IplImage *, IplImage *);
int thresh = 50, N = 11;

int searchMax(int a[], int n)
{
	int i,maxTag,max=0;
    for(i=0;i<n-5;i++){
        if(a[i]+a[i+1]+a[i+2]+a[i+3]+a[i+4]>max){
            max=a[i]+a[i+1]+a[i+2]+a[i+3]+a[i+4];
            maxTag=i;
        }
    }
    return maxTag;
}

/** @function main */
int main( int argc, char** argv )
{
    
 /// Load 图像
//    src = imread("/Users/yrguo/Desktop/pic/card1.jpg");
//src = imread("/Users/yrguo/Desktop/pic/card7.jpg");
//    src = imread("/Users/yrguo/Desktop/pic/card3.jpg");
//    src = imread("/Users/yrguo/Desktop/pic/card4.jpg");
    src = imread("/Users/yrguo/Desktop/pic/card10.png");
//    src = imread("/Users/yrguo/Desktop/pic/card15.png");
//    src = imread("/Users/yrguo/Desktop/pic/test.png");
//    cvtColor(src,src,CV_BGR2GRAY);
    if( !src.data )
    { return -1; }
    findPosition( 0, 0 );
    waitKey(0);
    return 0;
}
int func_nc8(int *b)
//端点的连通性检测
{
    int n_odd[4] = { 1, 3, 5, 7 };  //四邻域
    int i, j, sum, d[10];
    
    for (i = 0; i <= 9; i++) {
        j = i;
        if (i == 9) j = 1;
        if (abs(*(b + j)) == 1)
        {
            d[i] = 1;
        }
        else
        {
            d[i] = 0;
        }
    }
    sum = 0;
    for (i = 0; i < 4; i++)
    {
        j = n_odd[i];
        sum = sum + d[j] - d[j] * d[j + 1] * d[j + 2];
    }
    return (sum);
}

void cvHilditchThin(Mat src, Mat dst)
{
    if(src.type()!=CV_8UC1)
    {
        printf("只能处理二值或灰度图像\n");
        return;
    }
    //非原地操作时候，copy src到dst
    if(dst.data!=src.data)
    {
        src.copyTo(dst);
    }
    
    //8邻域的偏移量
    int offset[9][2] = {{0,0},{1,0},{1,-1},{0,-1},{-1,-1},
        {-1,0},{-1,1},{0,1},{1,1} };
    //四邻域的偏移量
    int n_odd[4] = { 1, 3, 5, 7 };
    int px, py;
    int b[9];                      //3*3格子的灰度信息
    int condition[6];              //1-6个条件是否满足
    int counter;                   //移去像素的数量
    int i, x, y, copy, sum;
    
    uchar* img;
    int width, height;
    width = dst.cols;
    height = dst.rows;
    img = dst.data;
    int step = dst.step ;
    do
    {
        
        counter = 0;
        
        for (y = 0; y < height; y++)
        {
            
            for (x = 0; x < width; x++)
            {
                
                //前面标记为删除的像素，我们置其相应邻域值为-1
                for (i = 0; i < 9; i++)
                {
                    b[i] = 0;
                    px = x + offset[i][0];
                    py = y + offset[i][1];
                    if (px >= 0 && px < width &&    py >= 0 && py <height)
                    {
                        // printf("%d\n", img[py*step+px]);
                        if (img[py*step+px] == 255)
                        {
                            b[i] = 1;
                        }
                        else if (img[py*step+px]  == 128)
                        {
                            b[i] = -1;
                        }
                    }
                }
                for (i = 0; i < 6; i++)
                {
                    condition[i] = 0;
                }
                
                //条件1，是前景点
                if (b[0] == 1) condition[0] = 1;
                
                //条件2，是边界点
                sum = 0;
                for (i = 0; i < 4; i++)
                {
                    sum = sum + 1 - abs(b[n_odd[i]]);
                }
                if (sum >= 1) condition[1] = 1;
                
                //条件3， 端点不能删除
                sum = 0;
                for (i = 1; i <= 8; i++)
                {
                    sum = sum + abs(b[i]);
                }
                if (sum >= 2) condition[2] = 1;
                
                //条件4， 孤立点不能删除
                sum = 0;
                for (i = 1; i <= 8; i++)
                {
                    if (b[i] == 1) sum++;
                }
                if (sum >= 1) condition[3] = 1;
                
                //条件5， 连通性检测
                if (func_nc8(b) == 1) condition[4] = 1;
                
                //条件6，宽度为2的骨架只能删除1边
                sum = 0;
                for (i = 1; i <= 8; i++)
                {
                    if (b[i] != -1)
                    {
                        sum++;
                    } else
                    {
                        copy = b[i];
                        b[i] = 0;
                        if (func_nc8(b) == 1) sum++;
                        b[i] = copy;
                    }
                }
                if (sum == 8) condition[5] = 1;
                
                if (condition[0] && condition[1] && condition[2] &&condition[3] && condition[4] && condition[5])
                {
                    img[y*step+x] = 128; //可以删除，置位GRAY，GRAY是删除标记，但该信息对后面像素的判断有用
                    counter++;
                    //printf("----------------------------------------------\n");
                    //PrintMat(dst);
                }
            }
        }
        
        if (counter != 0)
        {
            for (y = 0; y < height; y++)
            {
                for (x = 0; x < width; x++)
                {
                    if (img[y*step+x] == 128)
                        img[y*step+x] = 0;
                    
                }
            }
        }
        
    }while (counter != 0);
    
}
void printBinary(const char *title,  unsigned char*value, int length) {
    printf("\n-----%s[%d]-----\n", title, length);
    //    return;
    int printLen = 0;
    for (int i = 0; i < length; i++) {
        printLen++;
        printf("%02x", value[i]);
        if (printLen%2 == 0) {
            printf("  ");
        } else if (printLen%8==0) {
            printf("\n");
        }
    }
    printf("\n-----%s-----\n", title);
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
//    cvShowImage("Histogram", hHistImag);
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

void hisRow(Mat source,char * name){
//    IplImage *ipl_img = new IplImage(source);
//    IplImage *uImage = cvCreateImage(cvSize(ipl_img->width, 255), IPL_DEPTH_8U, 1);
//    cvResize(ipl_img, uImage,CV_INTER_LINEAR);
//    IplImage *hHistImag = cvCreateImage(cvSize(ipl_img->width, 255), IPL_DEPTH_8U, 1);
//    cvZero(hHistImag);
//    int i,j;
//    int HIS_COUNT = 200;
//    float HIS_WIDTH = ipl_img->width/200.0f;
//    int hSlot[HIS_COUNT];
//    memset(hSlot, 0, sizeof(int)*HIS_COUNT);
//    for(i=0;i<source.rows;i++){
//        for(j=0;j<source.cols;j++){
//            int tmp=cvGet2D(uImage,i,j).val[0];
////            if(tmp>0){
//                hSlot[(int)(j/HIS_WIDTH)]+=tmp;
////            }
//        }
//    }
//    for (i=0; i<HIS_COUNT; i++) {
////        if(hSlot[i]>source.rows/10.0){
//            cvRectangle(hHistImag, cvPoint(i*HIS_WIDTH,255-hSlot[i]/HIS_WIDTH), cvPoint((i+1)*HIS_WIDTH,255), cvScalar(255,0,0,0));
////        }
//    }
//    cvShowImage("Histogram", hHistImag);
    
    IplImage * src=new IplImage(source);
    //	cvSmooth(src,src,CV_BLUR,3,3,0,0);
	cvThreshold(src,src,50,255,CV_THRESH_BINARY_INV);
	IplImage* paintx=cvCreateImage( cvGetSize(src),IPL_DEPTH_8U, 1 );
	cvZero(paintx);
	int* v=new int[src->width];
	int* h=new int[src->height];
	memset(v,0,src->width*4);
	memset(h,0,src->height*4);
	
	int x,y;
	CvScalar s,t;
	for(x=0;x<src->width;x++)
	{
		for(y=0;y<src->height;y++)
		{
			s=cvGet2D(src,y,x);
			if(s.val[0]==0)
				v[x]++;
		}
	}
	
	for(x=0;x<src->width;x++)
	{
		for(y=0;y<v[x];y++)
		{
			t.val[0]=255;
			cvSet2D(paintx,y,x,t);
		}
	}
    cvShowImage(name, paintx);
}
void findRect(Mat imgMat,char *  str){
    char name[256]={0};
    sprintf(name, "findRect_%s",str);
    Mat drawing = Mat::zeros( imgMat.size(), CV_8UC3 );
    vector<vector<Point> > roi_contours;
    vector<Vec4i> roi_hierarchy;
    findContours(imgMat, roi_contours, roi_hierarchy, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );
    vector<Rect> roi_boundRect( roi_contours.size() );
    for( int i = 0; i < roi_contours.size(); i++ ){
        Rect r0= boundingRect(Mat(roi_contours[i]));//boundingRect获取这个外接矩形
        //        if((r0.width>=imgMat.cols/40.0f&&r0.height>=imgMat.rows/2.0f)||(r0.y<=imgMat.rows*0.4f&&r0.height>=imgMat.rows*0.6f)){
        rectangle(drawing,r0,Scalar(255,255,255),1);
        //        }
    }
    imshow(name,drawing);
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
        cvKMeans2(samples,nCuster,clusters,cvTermCriteria(CV_TERMCRIT_ITER,100,1.0));//开始聚类，迭代100次，终止误差1.0
        
        
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
void diaoke(Mat src){
    Mat img1(src.size(),CV_8UC3);
	for (int y=1; y<src.rows-1; y++)
	{
		uchar *p0 = src.ptr<uchar>(y);
		uchar *p1 = src.ptr<uchar>(y+1);
		uchar *q1 = img1.ptr<uchar>(y);
		for (int x=1; x<src.cols-1; x++)
		{
			for (int i=0; i<3; i++)
			{
				int tmp1 = p0[3*(x-1)+i]-p1[3*(x+1)+i]+128;//雕刻
				if (tmp1<0)
					q1[3*x+i]=0;
				else if(tmp1>255)
					q1[3*x+i]=255;
				else
					q1[3*x+i]=tmp1;
			}
		}
	}
//    cvtColor(img1,img1,CV_BGR2GRAY);
//    cvtColor(img1,img1,CV_BGR2Luv);
//        vector<Mat> rgb_planes;
//        split( img1, rgb_planes );
//        imshow("Image_L",rgb_planes[0]);
//        imshow("Image_A",rgb_planes[1]);
//        imshow("Image_B",rgb_planes[2]);
//    threshold(img1, img1, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
	imshow("diaoke",img1);

}
void fudiao(Mat rgbMat,Mat oriMat){
    
//    vector<Mat> rgb_planes;
//    split( oriMat, rgb_planes );
//    imshow("Image_L",rgb_planes[0]);
//    imshow("Image_A",rgb_planes[1]);
//    imshow("Image_B",rgb_planes[2]);
    
    
    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    int width=rgbMat.cols;
    int heigh=rgbMat.rows;
    Mat gray0,gray1,grag2;
    //去色
    cvtColor(rgbMat,gray0,CV_BGR2GRAY);
    gray0 = rgbMat.clone();
    //反色
    addWeighted(gray0,-1,NULL,0,255,gray1);
    //高斯模糊,高斯核的Size与最后的效果有关
    Mat img(rgbMat.size(),CV_8UC1);
    GaussianBlur(gray1,gray1,Size(13,13),0);
    for (int y=0; y<heigh; y++)
    {
        
        uchar* P0  = gray0.ptr<uchar>(y);
        uchar* P1  = gray1.ptr<uchar>(y);
        uchar* P  = img.ptr<uchar>(y);
        for (int x=0; x<width; x++)
        {
            int tmp0=P0[x];
            int tmp1=P1[x];
//            int diff = abs(tmp0 - tmp1);
//            if (diff >= 50)
//            {
//                P[x] = 0;
//            }
//            else
//            {
//                P[x] = 255;
//            }
            P[x] =(uchar) min((tmp0+(tmp0*tmp1)/(256-tmp1)),255);
        }
        
    }
    
//    cv::filter2D(img, img, img.depth(), kernel);
//    cv::filter2D(img, img, img.depth(), kernel);
//    threshold(img, img, 0, 255, CV_THRESH_BINARY);

    
    erode(img,img,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
//    erode(img,img,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
    dilate(img,img,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
//    erode(img,img,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
    threshold(img, img, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
//    erode(img,img,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
//    adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,3,5);
   
    
    
    IplImage *img_gray = new IplImage(img);
    IplImage* img_Clone=cvCloneImage(img_gray);
//    cvSmooth(img_gray,img_Clone,CV_BLUR,3,3,0,0);
//    cvDilate(img_Clone, img_Clone);
    Mat imgMat(img_Clone);
    
    imshow("sumiao2",imgMat);
    
    Mat drawing = Mat::zeros( imgMat.size(), CV_8UC3 );
    vector<vector<Point> > roi_contours;
    vector<Vec4i> roi_hierarchy;
    findContours(imgMat, roi_contours, roi_hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    vector<Rect> roi_boundRect( roi_contours.size() );
    for( int i = 0; i < roi_contours.size(); i++ ){
        Rect r0= boundingRect(Mat(roi_contours[i]));//boundingRect获取这个外接矩形
//        if((r0.width>=imgMat.cols/40.0f&&r0.height>=imgMat.rows/2.0f)||(r0.y<=imgMat.rows*0.4f&&r0.height>=imgMat.rows*0.6f)){
            rectangle(drawing,r0,Scalar(255,255,255),1);
//        }
    }
    imshow("sumiao3",drawing);

//    IplImage *img_gray2 = new IplImage(img);
//    IplImage* img_Clone2=cvCloneImage(img_gray);
//    cvSmooth(img_gray2,img_Clone2,CV_BLUR,3,3,0,0);
//    Mat imgMat2(img_Clone2);
//    erode(imgMat2,imgMat2,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
//    erode(imgMat2,imgMat2,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
//    imshow("sumiao",imgMat2);
    
    
    
//    Mat image_gray;
//    cvtColor(rgbMat, image_gray, CV_RGB2GRAY);
//    // Gradients in X and Y directions
//    Mat grad_x, grad_y;
//    Scharr(image_gray, grad_x, CV_32F, 1, 0);
//    Scharr(image_gray, grad_y, CV_32F, 0, 1);
//    // Calculate overall gradient
//    pow(grad_x, 2, grad_x);
//    pow(grad_y, 2, grad_y);
//    Mat grad = grad_x + grad_y;
//    sqrt(grad, grad);
//    // Convert to 8 bit depth for displaying
//    Mat edges;
//    grad.convertTo(edges, CV_8U);
//    addWeighted(edges,-1,NULL,0,255,edges);
//    
//    Mat element = getStructuringElement( MORPH_RECT,
//                                        Size( 3, 3 ),
//                                        Point( -1, -1 ) );
//    imshow("scharr",edges);
    
    
    
    
}
void rgbHis(Mat rgbMat){
    int bins = 256;
	int hist_size[] = {bins};
	float range[] = { 0, 256 };
	const float* ranges[] = { range};
	MatND hist_r,hist_g,hist_b;
	int channels_r[] = {0};
    
	calcHist( &rgbMat, 1, channels_r, Mat(), // do not use mask
             hist_r, 1, hist_size, ranges,
             true, // the histogram is uniform
             false );
    
	int channels_g[] = {1};
	calcHist( &rgbMat, 1, channels_g, Mat(), // do not use mask
             hist_g, 1, hist_size, ranges,
             true, // the histogram is uniform
             false );
    
	int channels_b[] = {2};
	calcHist( &rgbMat, 1, channels_b, Mat(), // do not use mask
             hist_b, 1, hist_size, ranges,
             true, // the histogram is uniform
             false );
	double max_val_r,max_val_g,max_val_b;
	minMaxLoc(hist_r, 0, &max_val_r, 0, 0);
	minMaxLoc(hist_g, 0, &max_val_g, 0, 0);
	minMaxLoc(hist_b, 0, &max_val_b, 0, 0);
	int scale = 1;
	int hist_height=256;
	Mat hist_img = Mat::zeros(hist_height,bins*3, CV_8UC3);
	for(int i=0;i<bins;i++)
	{
		float bin_val_r = hist_r.at<float>(i);
		float bin_val_g = hist_g.at<float>(i);
		float bin_val_b = hist_b.at<float>(i);
		int intensity_r = cvRound(bin_val_r*hist_height/max_val_r);  //要绘制的高度
		int intensity_g = cvRound(bin_val_g*hist_height/max_val_g);  //要绘制的高度
		int intensity_b = cvRound(bin_val_b*hist_height/max_val_b);  //要绘制的高度
		rectangle(hist_img,Point(i*scale,hist_height-1),
                  Point((i+1)*scale - 1, hist_height - intensity_r),
                  CV_RGB(255,0,0));
        
		rectangle(hist_img,Point((i+bins)*scale,hist_height-1),
                  Point((i+bins+1)*scale - 1, hist_height - intensity_g),
                  CV_RGB(0,255,0));
        
		rectangle(hist_img,Point((i+bins*2)*scale,hist_height-1),
                  Point((i+bins*2+1)*scale - 1, hist_height - intensity_b),
                  CV_RGB(0,0,255));
        
	}
	imshow( "RGB Histogram", hist_img );
}

void colorHis(Mat roi){
    IplImage * src= new IplImage(roi);
    IplImage* hsv = cvCreateImage( cvGetSize(src), 8, 3 );
	IplImage* h_plane = cvCreateImage( cvGetSize(src), 8, 1 );
	IplImage* s_plane = cvCreateImage( cvGetSize(src), 8, 1 );
	IplImage* v_plane = cvCreateImage( cvGetSize(src), 8, 1 );
	IplImage* planes[] = { h_plane, s_plane };
    
	/** H 分量划分为16个等级，S分量划分为8个等级 */
	int h_bins = 16, s_bins = 8;
	int hist_size[] = {h_bins, s_bins};
    
	/** H 分量的变化范围 */
	float h_ranges[] = { 0, 180 };
    
	/** S 分量的变化范围*/
	float s_ranges[] = { 0, 255 };
	float* ranges[] = { h_ranges, s_ranges };
    
	/** 输入图像转换到HSV颜色空间 */
	cvCvtColor( src, hsv, CV_BGR2HSV );
	cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );
    
	/** 创建直方图，二维, 每个维度上均分 */
	CvHistogram * hist = cvCreateHist( 2, hist_size, CV_HIST_ARRAY, ranges, 1 );
	/** 根据H,S两个平面数据统计直方图 */
	cvCalcHist( planes, hist, 0, 0 );
    
	/** 获取直方图统计的最大值，用于动态显示直方图 */
	float max_value;
	cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
    
    
	/** 设置直方图显示图像 */
	int height = 240;
	int width = (h_bins*s_bins*6);
	IplImage* hist_img = cvCreateImage( cvSize(width,height), 8, 3 );
	cvZero( hist_img );
    
	/** 用来进行HSV到RGB颜色转换的临时单位图像 */
	IplImage * hsv_color = cvCreateImage(cvSize(1,1),8,3);
	IplImage * rgb_color = cvCreateImage(cvSize(1,1),8,3);
	int bin_w = width / (h_bins * s_bins);
	for(int h = 0; h < h_bins; h++)
	{
		for(int s = 0; s < s_bins; s++)
		{
			int i = h*s_bins + s;
			/** 获得直方图中的统计次数，计算显示在图像中的高度 */
			float bin_val = cvQueryHistValue_2D( hist, h, s );
			int intensity = cvRound(bin_val*height/max_value);
            
			/** 获得当前直方图代表的颜色，转换成RGB用于绘制 */
			cvSet2D(hsv_color,0,0,cvScalar(h*180.f / h_bins,s*255.f/s_bins,255,0));
			cvCvtColor(hsv_color,rgb_color,CV_HSV2BGR);
			CvScalar color = cvGet2D(rgb_color,0,0);
            
			cvRectangle( hist_img, cvPoint(i*bin_w,height),
                        cvPoint((i+1)*bin_w,height - intensity),
                        color, -1, 8, 0 );
		}
	}
    
    
	cvShowImage( "H-S Histogram", hist_img );
    
    
}
void toPics(Mat roi_img){
    int erosion_type = MORPH_RECT;
    Mat roi_img_dst;
    Mat roi_element = getStructuringElement( erosion_type,
                                            Size( 3, 3 ),
                                            Point( -1, -1 ) );
    
    
    Mat roi_img_dst_gray;
    
//    cvtColor(roi_img,roi_img_dst,CV_BGR2HSV);
    cvtColor(roi_img,roi_img_dst,CV_BGR2Lab);
//    cvtColor(roi_img,roi_img_dst,CV_BGR2HLS);
//    cvtColor(roi_img,roi_img_dst,CV_BGR2Luv);
//    cvtColor(roi_img,roi_img_dst,CV_BGR2YCrCb);
//    cvtColor(roi_img,roi_img_dst,CV_BGR2YUV);
    
    
        vector<Mat> rgb_planes;
        split( roi_img_dst, rgb_planes );
        imshow("Image_L",rgb_planes[0]);
        imshow("Image_A",rgb_planes[1]);
        imshow("Image_B",rgb_planes[2]);
//    cvtColor(rgb_planes[2],roi_img_dst_gray,CV_BGR2GRAY);
    threshold(rgb_planes[2], roi_img_dst_gray, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
//    dilate(roi_img_dst_gray,roi_img_dst_gray,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( 0, 0 ) ));
    imshow("gray",roi_img_dst_gray);
    findRect(roi_img_dst_gray, "grag_rect");
    IplImage* single = toSinglePic(roi_img_dst,20);
    Mat singleMat = Mat(single);
    medianBlur(singleMat, singleMat, 1);
    Mat drawing = Mat::zeros( roi_img_dst.size(), CV_8UC3 );
    vector<vector<Point> > roi_contours;
    vector<Vec4i> roi_hierarchy;
    findContours(singleMat, roi_contours, roi_hierarchy, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    vector<Rect> roi_boundRect( roi_contours.size() );
    for( int i = 0; i < roi_contours.size(); i++ ){
        Rect r0= boundingRect(Mat(roi_contours[i]));//boundingRect获取这个外接矩形
        if((r0.width>=roi_img_dst.cols/40.0f&&r0.height>=roi_img_dst.rows/2.0f)||(r0.y<=roi_img_dst.rows*0.4f&&r0.height>=roi_img_dst.rows*0.6f)){
            rectangle(drawing,r0,Scalar(255,255,255),1);
            if(r0.width>r0.height){
                Mat pic,pic_org;
                r0.y=0;
                r0.height=roi_img_dst.rows;
                roi_img(r0).copyTo(pic);
                
                char str1[60] = {0};
                sprintf(str1,"/Users/yrguo/Desktop/pic/roi_img_%d.jpg",i);
                vector<int> comp;
                comp.push_back(CV_IMWRITE_JPEG_QUALITY);
                comp.push_back(100);
                imwrite(str1,pic,comp);
//                cvHilditchThin(pic, pic_org);
//                 medianBlur(pic, pic_org, 1);
                cvtColor(pic,pic_org,CV_BGR2GRAY);
                threshold(pic_org, pic_org, 75, 255, CV_THRESH_BINARY_INV);
//                medianBlur(pic_org, pic_org, 5);
//                Canny(pic_org, pic_org, 5, 15);
//                char str2[60] = {0};
//                sprintf(str2,"roi_img_%d_his",i);
//                hisRow(pic_org,str2);
//                adaptiveThreshold(pic_org, pic_org, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,3,5);
//                GaussianBlur(pic_org,pic_org,Size(3,3),0);
//                dilate(pic_org,pic_org,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( -1, -1 ) ));
//                erode(pic_org,pic_org,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( 0, 0 ) ));
                
//                Mat kernel(3,3,CV_32F,Scalar(-1));
//                kernel.at<float>(1,1) = 8;
//                filter2D(pic_org,pic_org,pic_org.depth(),kernel);
                
                
//                threshold(pic_org, pic_org, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
//                morphologyEx(pic_org, pic_org, MORPH_OPEN, getStructuringElement(MORPH_RECT,Size( 1,1),Point( -1,-1)));
//                morphologyEx(pic_org, pic_org, MORPH_CLOSE, getStructuringElement(MORPH_RECT,Size( 1,1),Point( -1,-1)));
//                
//                morphologyEx(pic_org, pic_org, MORPH_OPEN, getStructuringElement(MORPH_RECT,Size( 1,1),Point( -1,-1)));
//                cvHilditchThin(pic_org, pic_org);
//                dilate(pic_org,pic_org,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( 0, 0 ) ));
//                erode(pic_org,pic_org,getStructuringElement(MORPH_RECT,Size( 3, 3 ),Point( 0, 0 ) ));
                
//                GaussianBlur(pic_org,pic_org,Size(1,1),0);
                imshow(str1,pic_org);
//                colorHis(pic);
//                rgbHis(pic);
                findRect(pic_org,str1);
                
            }else{
                Mat pic,pic_org;
                r0.y=0;
                r0.height=roi_img_dst.rows;
                roi_img(r0).copyTo(pic);
                char str1[60] = {0};
                sprintf(str1,"/Users/yrguo/Desktop/pic/roi_img_%d.jpg",i);
                vector<int> comp;
                comp.push_back(CV_IMWRITE_JPEG_QUALITY);
                comp.push_back(100);
                imwrite(str1,pic,comp);
            }
        }
    }
    imshow("out",drawing);
    
    //    colorHis(roi_img_dst);
    
    //     GaussianBlur( roi_img_dst, roi_img_dst, Size(2*erosion_size + 1, 2*erosion_size + 1 ), 0, 0, BORDER_DEFAULT );
    //    erode( roi_img_dst, roi_img_dst, roi_element);
    //    dilate( roi_img_dst, roi_img_dst, roi_element );
    //    erode( roi_img_dst, roi_img_dst, roi_element);
    //    dilate( roi_img_dst, roi_img_dst, roi_element );
    
    
    //    Canny(roi_img_dst, roi_img_dst, 50, 150,7);
    
//    fudiao(roi_img_dst,roi_img);
    
    
    //    rgbHis(roi_img_dst);
    //
    //    Mat roi_grad_x, roi_grad_y;
    //    Mat roi_abs_grad_x, roi_abs_grad_y;
    //
    //    /// 求 X方向梯度
    //    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    //    Sobel( roi_img_dst, roi_grad_x, CV_8U, 1, 0, 1, 0, 0, BORDER_DEFAULT );
    //    convertScaleAbs( roi_grad_x, roi_abs_grad_x );
    //
    //    /// 求Y方向梯度
    //    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    //    Sobel( roi_img_dst, roi_grad_y, CV_8U, 0, 1, 1, 1, 0, BORDER_DEFAULT );
    //    convertScaleAbs( roi_grad_y, roi_abs_grad_y );
    //
    //    /// 合并梯度(近似)
    //    addWeighted( roi_abs_grad_x, 0, roi_abs_grad_y, 1, 0, roi_img_dst );
    //
    //
    //
    //
    //    threshold(roi_img_dst, roi_img_dst, 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
    
//    vector<int> comp;
//	comp.push_back(CV_IMWRITE_JPEG_QUALITY);
//	comp.push_back(98);
//	imwrite("/Users/yrguo/Desktop/pic/out.jpg",roi_img_dst,comp);
    
    imshow( "roi_img_dst Demo", roi_img_dst );
    moveWindow("roi_img_dst Demo", roi_img_dst.cols, 0);
}
void findPosition( int, void* )
{
    int erosion_type;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else { erosion_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( erosion_type,
                                        Size( 2*erosion_size + 1, 2*erosion_size + 1 ),
                                        Point( -1, -1 ) );
//
//    /// 腐蚀操作
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
    
//    imshow( "Dilation Demo", erosion_dst );
    
    vector<vector<Point> > squares;
//    findSquares(erosion_dst, squares);
//    drawSquares(erosion_dst, squares);
    
    
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
//
//    RNG rng(12345);
//    
//    /// 画多边形轮廓 + 包围的矩形框 + 圆形框
    Mat drawing = Mat::zeros( erosion_dst.size(), CV_8UC3 );
//    imshow( "Erosion Demo", drawing );
    
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
    Mat roi_img;
    src(rect).copyTo(roi_img);
    imshow("roi_img",roi_img);
    vector<int> comp;
    comp.push_back(CV_IMWRITE_JPEG_QUALITY);
    comp.push_back(100);
    char str1[60] = {0};
    sprintf(str1,"/Users/yrguo/Desktop/pic/roi_img.jpg");
    imwrite(str1,roi_img,comp);
    toPics(roi_img);
}


void kirsch(IplImage *src,IplImage *dst)
{
    dst = cvCloneImage(src);
    //cvConvert(src,srcMat); //
    int x,y;
    float a,b,c,d;
    float p1,p2,p3,p4,p5,p6,p7,p8,p9;
    uchar* ps = (uchar*)src->imageData ; //ps
    uchar* pd = (uchar*)dst->imageData ; //pd
    int w = dst->width;
    int h = dst->height;
    int step = dst->widthStep;
    
    for(x = 0;x<w-2;x++)      //?x+1?y+1)9  1 4 7
    {                                                            // 2 5 8
        for(y = 0;y<h-2;y++)                                     // 3 6 9
        {
            p1=ps[y*step+x];
            p2=ps[y*step+(x+1)];
            p3=ps[y*step+(x+2)];
            p4=ps[(y+1)*step+x];
            p5=ps[(y+1)*step+(x+1)];
            p6=ps[(y+1)*step+(x+2)];
            p7=ps[(y+2)*step+x];
            p8=ps[(y+2)*step+(x+1)];
            p9=ps[(y+2)*step+(x+2)];//(i+1,j+1)
            
            a = fabs(float(-5*p1-5*p2-5*p3+3*p4+3*p6+3*p7+3*p8+3*p9));    //4
            b = fabs(float(3*p1-5*p2-5*p3+3*p4-5*p6+3*p7+3*p8+3*p9));
            c = fabs(float(3*p1+3*p2-5*p3+3*p4-5*p6+3*p7+3*p8-5*p9));
            d = fabs(float(3*p1+3*p2+3*p3+3*p4-5*p6+3*p7-5*p8-5*p9));
            a = max(a,b);                                         //
            a = max(a,c);
            a = max(a,d);
            pd[(y+1)*step+(x+1)] = a;
            /*  if(a>100)
             {
             pd[(y+1)*step+(x+1)]=255;
             }
             else pd[(y+1)*step+(x+1)]=0;*/
        }
    }
//    double min_val = 0, max_val = 0;//
//    cvMinMaxLoc(dst,&min_val,&max_val);
//    printf("max_val = %f\nmin_val = %f\n",max_val,min_val);
    
    cvNormalize(dst,dst,0,255,CV_MINMAX); //
    cvSaveImage("KirschImg.jpg", dst);//
    cvNamedWindow("kirsch",1);
    cvShowImage("kirsch",dst);
}
/** @function Dilation */
void Dilation( int, void* )
{
    int dilation_type;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else { dilation_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( dilation_type,
                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        Point( dilation_size, dilation_size ) );
    ///膨胀操作
    dilate( src, dilation_dst, element );
    imshow( "Dilation Demo", dilation_dst );
}