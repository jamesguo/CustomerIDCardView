#include <iostream> 
#include <fstream>
using namespace std;

struct SimpleClassifier
{
	REAL thresh;
	REAL error;
	int parity;
	int type;
	int x1,x2,x3,x4,y1,y2,y3,y4;

	inline const REAL GetOneFeature(const IntImage& im) const;
	inline const REAL GetOneFeatureTranslation(REAL** const data,const int y) const;
	inline const int Apply(const REAL value) const;
	inline const int Apply(const IntImage& im) const;
	
	void ReadFromBin(ifstream& fout);
};

const int SimpleClassifier::Apply(const IntImage& im) const
{
	if(parity == 1)
		return (GetOneFeature(im)<thresh)?1:0;
	else
		return (GetOneFeature(im)>=thresh)?1:0;
}

const int SimpleClassifier::Apply(const REAL value) const
{
	if(parity == 1)
		return (value<thresh)?1:0;
	else
		return (value>=thresh)?1:0;
}

const REAL SimpleClassifier::GetOneFeature(const IntImage& im) const
{
	REAL f1;
	REAL** data = im.data;

	switch(type)
	{
	case 0:
		f1 =   data[x1][y3] - data[x1][y1] + data[x3][y3] - data[x3][y1]
			 + 2*(data[x2][y1] - data[x2][y3]);
		break;
	case 1:
		f1 =   data[x3][y1] + data[x3][y3] - data[x1][y1] - data[x1][y3]
			 + 2*(data[x1][y2] - data[x3][y2]);
		break;
	case 2:
		f1 =   data[x1][y1] -data[x1][y3] + data[x4][y3] - data[x4][y1]
			 + 3*(data[x2][y3] - data[x2][y1] + data[x3][y1] - data[x3][y3]);
		break;
	case 3:
		f1 =   data[x1][y1] - data[x1][y4] + data[x3][y4] - data[x3][y1]
			 + 3*(data[x3][y2] - data[x3][y3] + data[x1][y3] - data[x1][y2] );
		break;
	case 4:
		f1 =   data[x1][y1] + data[x1][y3] + data[x3][y1] + data[x3][y3]
			 - 2*(data[x2][y1] + data[x2][y3] + data[x1][y2] + data[x3][y2])
			 + 4*data[x2][y2];
		break;
	}
	return f1/im.variance;
}
