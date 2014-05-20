#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double REAL;
#else
typedef float REAL;
#endif



class IntImage
{
public:
	IntImage();
	~IntImage();

	void Clear(void); 
	void SetSize(const CvSize size);
	IntImage& operator=(const IntImage& source);
	void CalcSquareAndIntegral(IntImage& square,IntImage& image) const;
	void CalculateVarianceAndIntegralImageInPlace(void);
	void Resize(IntImage &result,  REAL ratio) const;
	void Copy(const IntImage& source);
	void Load(const char *& filename);
	void Save(const char *& filename) const;

public:
	int height; 
	int width;  
	REAL** data; 
					
	REAL* buf;   
	REAL variance;
	int label;
};

void SwapIntImage(IntImage& i1,IntImage& i2);