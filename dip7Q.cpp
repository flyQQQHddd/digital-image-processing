#include <iostream>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <Windows.h>
using namespace std;
using namespace cv;


//***************************************函数声明***************************************************
//**************************************************************************************************




//---------------------------------------灰度变换函数-----------------------------------------------

namespace gt
{

	void greyTransformation(
		const Mat& inImage,
		Mat& outImage,
		Mat& TMat,
		void (*transFun)(const uchar& inValue, uchar& outValue, void* reservedWord),
		void* reservedWord = NULL);

	inline void copyValue(const uchar& inValue, uchar& outValue, void* reservedWord);
	inline void linearTrans(const uchar& inValue, uchar& outValue, void* reservedWor);
	inline void logTrans(const uchar& inValue, uchar& outValuet, void* reservedWord);
	inline void piecewiseLinearTrans(const uchar& inValue, uchar& outValue, void* reservedWord);

}

//---------------------------------------空间域滤波函数---------------------------------------------

namespace sf
{

	enum OperType { Lowpass, HighPass };
	class Operator {
	public:
		Operator(OperType type);
		Operator();
		vector<double>m_value;
		unsigned int m_size;
	};
	

	void linearFilter(const Mat& inImage, Mat& outImage, const Operator& oper);
	void midFilter(const Mat& inImage, Mat& outImage, int size);
	void fastMidFilter(const Mat& inImage, Mat& outImage, unsigned int diameter);

}

//---------------------------------------几何变换函数-----------------------------------------------

namespace ge
{

	class TransMat {

	public:

		TransMat();
		vector<vector<double>>m_value;

		TransMat& SetTranslation(double p, double q);
		TransMat& SetScaling(double a, double b);
		TransMat& SetRotation(double ang);

		bool IsZeros();
		void Clear();
		vector<vector<double>> GetTranspose();

	};

	void geometricTrans(const Mat& inImage, Mat& outImage, Mat& outImagePart, TransMat& transMat);

	void pointTrans(const Vec2d& inVec, Vec2d& outVal, vector<vector<double>>& T);


}

//---------------------------------------直方图匹配函数---------------------------------------------

namespace hs
{

	class Histogram {

	public:
		Histogram();
		Histogram(int i);

		vector<int>m_sum;
		vector<double>m_p;

		int m_all;

		void Draw(Mat& outImage);

	};

	void histogramStat(const Mat& inImage, Histogram& hist);

	void histogramMatching(const Histogram& src, const Histogram& dst, map<uchar, uchar>& index);

	void transHistogram(const Mat& inImage, Mat& outImage, map<uchar, uchar>& index);

}

//---------------------------------------交互函数---------------------------------------------------

namespace cw
{

	void multipleImage(vector<Mat> imgVector, Mat& dst, int imgCols);
	Mat readImage();
	void saveImage(Mat& mat);

}

//---------------------------------------菜单及色彩函数---------------------------------------------

void color(const unsigned short textColor);
int menu();

//---------------------------------------菜单出口宏-------------------------------------------------

//图像点运算：灰度变换
#define GRAY_LINEAR_TRANS            11//线性变换
#define GRAY_PIECEWISE_LINEAR_TRANS  12//分段线性变换
#define GRAY_LOG_TRANS               13//对数变换

//图像局部处理：高通滤波、低通滤波、中值滤波
#define HIGH_PASS_FILTER             21//高通滤波
#define LOW_PASS_FILTER              22//低通滤波
#define MID_FILTER                   23//中值滤波
#define DIY_FILTER                   24//自定义模板

//图像的几何处理：平移、缩放、旋转geometricTrans
#define GEOMETRIC_TRANS              3

//直方图匹配
#define HISTOGRAM_EQUALITION         41//直方图均衡化
#define HISTOGRAM_MATCHING           42//直方图规定化

//图像二值化：状态法及判断分析法
//#define 

//退出
#define END                          0//退出




//***************************************函数定义***************************************************
//**************************************************************************************************




//---------------------------------------灰度变换函数定义-------------------------------------------

void gt::greyTransformation(
	const Mat& inImage,
	Mat& outImage,
	Mat& TMat,
	void (*transFun)(const uchar& inValue, uchar& outValue, void* reservedWord),
	void* reservedWord)
{
	outImage.create(inImage.rows, inImage.cols, inImage.type());
	TMat = Mat::zeros(256, 256, CV_8UC1);


	map<uchar, uchar>index;
	for (int i = 0; i < 256; i++) {

		uchar Tr;
		transFun(i, Tr, reservedWord);
		index.insert(make_pair(i, Tr));
		TMat.at<uchar>(255 - Tr, i) = 255;

	}

	for (int i = 0; i < inImage.rows; i++) {

		for (int j = 0; j < inImage.cols; j++) {

			outImage.at<uchar>(i, j) = index[inImage.at<uchar>(i, j)];

		}

	}


}
inline void gt::copyValue(const uchar& inValue, uchar& outValue, void* reservedWord)
{
	//T(r)=r
	outValue = inValue;
}
inline void gt::linearTrans(const uchar& inValue, uchar& outValue, void* reservedWord)
{

	//T(r)=a*r+b
	if (reservedWord == NULL)return;

	int a = (*((Vec2i*)reservedWord))[0];
	int b = (*((Vec2i*)reservedWord))[1];
	int Tr = a * (int)inValue + b;


	if (Tr < 0) {
		outValue = (uchar)0;
	}
	else if (Tr > 255){
		outValue = (uchar)255;
	}
	else {
		outValue = (uchar)Tr;
	}
}
inline void gt::logTrans(const uchar& inValue, uchar& outValue, void* reservedWord)
{
	//T(r)=c*log(r+1)
	if (reservedWord == NULL)return;
	int c = *((int*)reservedWord);
	int Tr = c * std::log((int)inValue + 1);

	if (Tr < 0) {
		outValue = (uchar)0;
	}
	else if (Tr > 255) {
		outValue = (uchar)255;
	}
	else {
		outValue = (uchar)Tr;
	}
}
inline void gt::piecewiseLinearTrans(const uchar& inValue, uchar& outValue, void* reservedWord) {

	if (reservedWord == NULL)return;
	vector<Vec2i>points = *((vector<Vec2i>*)reservedWord);
	float a = points[0][0];
	float b = points[1][0];
	float c = points[0][1];
	float d = points[1][1];

	float Tr;

	if (inValue < a) {

		Tr = c / a * inValue;

	}
	else if (inValue < b) {

		Tr = ((d - c) / (b - a)) * (inValue - a) + c;

	}
	else {

		Tr = ((255 - d) / (255 - b)) * (inValue - b) + d;

	}

	if (Tr < 0) {
		outValue = (uchar)0;
	}
	else if (Tr > 255) {
		outValue = (uchar)255;
	}
	else {
		outValue = (uchar)Tr;
	}

}



//---------------------------------------空间域滤波函数定义-----------------------------------------

sf::Operator::Operator(OperType type) {

	switch (type) {
	case Lowpass:

		this->m_size = 3;
		this->m_value = vector<double>{ 1.0 / 9.0, 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0, };

		break;
	case HighPass:

		this->m_size = 3;
		//this->m_value = vector<double>{ -1,-1,-1,-1,9, -1,-1,-1,-1 };
		this->m_value = vector<double>{ 0,-1,0,-1,4, -1,0,-1,0 };

		break;
	}

}
sf::Operator::Operator() 
{
	this->m_value = vector<double>(9, 0);
}
void sf::linearFilter(const Mat& inImage, Mat& outImage, const Operator& oper)
{
	const unsigned int helfSize = oper.m_size / 2;
	const unsigned int num = oper.m_size * oper.m_size;

	outImage.create(inImage.rows, inImage.cols, inImage.type());

	for (int outi = 0; outi < inImage.rows; outi++) {

		for (int outj = 0; outj < inImage.cols; outj++) {

			int k = 0;

			double ans = 0.0f;

			for (int ini = outi - helfSize; ini <= outi + helfSize; ini++) {

				for (int inj = outj - helfSize; inj <= outj + helfSize; inj++) {

					if (ini >= 0 && ini < inImage.rows && inj >= 0 && inj < inImage.cols)ans += oper.m_value[k] * inImage.at<uchar>(ini, inj);

					k++;
				}

			}
			
			if (ans > 0)outImage.at<uchar>(outi, outj) = (uchar)ans;
			else outImage.at<uchar>(outi, outj) = 0;

		}

	}

}
void sf::midFilter(const Mat& inImage, Mat& outImage, int size)
{
	int helfSize = size / 2;
	int num = size * size;

	outImage.create(inImage.rows, inImage.cols, inImage.type());

	for (int outi = 0; outi < inImage.rows; outi++) {

		for (int outj = 0; outj < inImage.cols; outj++) {

			vector<uchar>val;

			int ini = outi - helfSize;
			for (; ini <= outi + helfSize; ini++) {

				int inj = outj - helfSize;
				for (; inj <= outj + helfSize; inj++) {

					if (ini >= 0 && ini < inImage.rows && inj >= 0 && inj < inImage.cols)val.push_back(inImage.at<uchar>(ini, inj));
					else val.push_back(0);

				}

			}

			for (int i = 0; i < val.size() - 1; i++) {

				for (int j = 0; j < val.size() - 1 - i; j++) {

					if (val[j] > val[j + 1]) {

						uchar temp = val[j];
						val[j] = val[j + 1];
						val[j + 1] = temp;

					}

				}

			}

			outImage.at<uchar>(outi, outj) = val[(int)(size * size / 2)];

		}

	}


}
void sf::fastMidFilter(const Mat& inImage, Mat& outImage, unsigned int diameter) {

	if (diameter < 3)
	{
		diameter = 3;
	}

	int row = inImage.rows;
	int col = inImage.cols;
	Mat dst(row, col, CV_8UC1);
	int Hist[256] = { 0 };
	int radius = (diameter - 1) / 2;
	int windowSize = diameter * diameter;
	int threshold = windowSize / 2 + 1;
	uchar* srcData = inImage.data;
	uchar* dstData = dst.data;
	int right = col - radius;
	int bot = row - radius;

	for (int j = radius; j < bot; j++) {
		for (int i = radius; i < right; i++) {
			//每一行第一个待滤波元素建立直方图
			if (i == radius) {
				memset(Hist, 0, sizeof(Hist));
				for (int y = j - radius; y <= min(j + radius, row); y++) {
					for (int x = i - radius; x <= min(i + radius, col); x++) {
						uchar val = srcData[y * col + x];
						Hist[val]++;
					}
				}
			}
			else {
				int L = i - radius - 1;
				int R = i + radius;
				for (int y = j - radius; y <= min(j + radius, row); y++) {
					//更新左边一列
					Hist[srcData[y * col + L]]--;
					//更新右边一列
					Hist[srcData[y * col + R]]++;
				}
			}


			//查找中值
			uchar medianVal;
			int sum = 0;
			int flag = 0;
			for (int i = 0; i < 256; i++) {
				sum += Hist[i];
				if (sum >= threshold) {
					medianVal = i;
					flag = 1;
					break;
				}
			}
			if (flag == 0)medianVal = 255;



			dstData[j * col + i] = medianVal;
		}
	}
	//边界直接赋值
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < radius; j++) {
			int id1 = j * col + i;
			int id2 = (row - j - 1) * col + i;
			dstData[id1] = srcData[id1];
			dstData[id2] = srcData[id2];
		}
	}
	for (int i = radius; i < row - radius - 1; i++) {
		for (int j = 0; j < radius; j++) {
			int id1 = i * col + j;
			int id2 = i * col + col - j - 1;
			dstData[id1] = srcData[id1];
			dstData[id2] = srcData[id2];
		}
	}

	outImage = dst;
}


//---------------------------------------几何变换函数定义-------------------------------------------

void ge::geometricTrans(const Mat& inImage, Mat& outImage, Mat& outImagePart, TransMat& transMat)
{
	
	//new=A*old

	Vec2d LU, LD, RU, RD;
	pointTrans(Vec2d(0, 0), LU, transMat.m_value);
	pointTrans(Vec2d(inImage.rows - 1, 0), LD, transMat.m_value);
	pointTrans(Vec2d(0, inImage.cols - 1), RU, transMat.m_value);
	pointTrans(Vec2d(inImage.rows - 1, inImage.cols - 1), RD, transMat.m_value);
	vector<double>x;
	vector<double>y;

	x.push_back(LU[0]);
	x.push_back(LD[0]);
	x.push_back(RU[0]);
	x.push_back(RD[0]);
	y.push_back(LU[1]);
	y.push_back(LD[1]);
	y.push_back(RU[1]);
	y.push_back(RD[1]);

	double minX = *min_element(x.begin(), x.end());
	double maxX = *max_element(x.begin(), x.end());
	double minY = *min_element(y.begin(), y.end());
	double maxY = *max_element(y.begin(), y.end());


	outImage.create(inImage.rows, inImage.cols, inImage.type());

	outImagePart.create(maxX - minX, maxY - minY, inImage.type());


	//T*new=old
	vector<vector<double>>T = transMat.GetTranspose();
	for (int i = 0; i < outImage.rows; i++) {

		for (int j = 0; j < outImage.cols; j++) {

			Vec2d det;
			pointTrans(Vec2d(i, j), det, T);
			if (det[0] < 0 || det[0] >= inImage.rows || det[1] < 0 || det[1] >= inImage.cols)outImage.at<uchar>(i, j) = 0;
			else outImage.at<uchar>(i, j) = inImage.at<uchar>(det[0], det[1]);

		}

	}

	for (int i = 0; i < outImagePart.rows; i++) {

		for (int j = 0; j < outImagePart.cols; j++) {

			Vec2d det;
			pointTrans(Vec2d(i + (int)minX, j + (int)minY), det, T);
			if (det[0] < 0 || det[0] >= inImage.rows || det[1] < 0 || det[1] >= inImage.cols)outImagePart.at<uchar>(i, j) = 0;
			else outImagePart.at<uchar>(i, j) = inImage.at<uchar>(det[0], det[1]);

		}

	}


}
void ge::pointTrans(const Vec2d& inVec, Vec2d& outVal, vector<vector<double>>& T) {

	outVal[0] = T[0][0] * inVec[0] + T[0][1] * inVec[1] + T[0][2];
	outVal[1] = T[1][0] * inVec[0] + T[1][1] * inVec[1] + T[1][2];

	//double z = T[2][0] * inVec[0] + T[2][1] * inVec[1] + T[2][2];

	//outVal[0] /= z;
	//outVal[1] /= z;

}
vector<vector<double>> ge::TransMat::GetTranspose()
{
	int N = 3;
	double max, temp;
	vector<vector<double>>inMatrix = this->m_value;
	// 定义一个临时矩阵t
	vector<vector<double>>t = inMatrix;
	//定义输出结果
	vector<vector<double>>outMatrix = inMatrix;
	// 初始化outMatrix矩阵为单位矩阵
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			outMatrix[i][j] = (i == j) ? (double)1 : 0;
		}
	}

	// 进行列主消元，找到每一列的主元
	for (int i = 0; i < N; i++)
	{
		max = t[i][i];
		// 用于记录每一列中的第几个元素为主元
		int k = i;
		// 寻找每一列中的主元元素
		for (int j = i + 1; j < N; j++)
		{
			if (fabs(t[j][i]) > fabs(max))
			{
				max = t[j][i];
				k = j;
			}
		}
		//cout<<"the max number is "<<max<<endl;
		// 如果主元所在的行不是第i行，则进行行交换
		if (k != i)
		{
			// 进行行交换
			for (int j = 0; j < N; j++)
			{
				temp = t[i][j];
				t[i][j] = t[k][j];
				t[k][j] = temp;
				// 伴随矩阵B也要进行行交换
				temp = outMatrix[i][j];
				outMatrix[i][j] = outMatrix[k][j];
				outMatrix[k][j] = temp;
			}
		}
		if (t[i][i] == 0)
		{
			cout << "\nthe matrix does not exist inverse matrix\n";
			break;
		}
		// 获取列主元素
		temp = t[i][i];
		// 将主元所在的行进行单位化处理
		//cout<<"\nthe temp is "<<temp<<endl;
		for (int j = 0; j < N; j++)
		{
			t[i][j] = t[i][j] / temp;
			outMatrix[i][j] = outMatrix[i][j] / temp;
		}
		for (int j = 0; j < N; j++)
		{
			if (j != i)
			{
				temp = t[j][i];
				//消去该列的其他元素
				for (int k = 0; k < N; k++)
				{
					t[j][k] = t[j][k] - temp * t[i][k];
					outMatrix[j][k] = outMatrix[j][k] - temp * outMatrix[i][k];
				}
			}

		}

	}

	return outMatrix;

}
ge::TransMat::TransMat() 
{
	this->m_value = vector<vector<double>>(3, { 0,0,0 });
	for (int i = 0; i < 3; i++) {

		m_value[i][i] = 1;

	}
}
ge::TransMat& ge::TransMat::SetTranslation(double p, double q)
{
	//做矩阵乘法
	TransMat mat;
	mat.m_value[0][2] = p;
	mat.m_value[1][2] = q;

	vector<vector<double>>res(3, { 0,0,0 });

	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 3; j++) {

			for (int num = 0; num < 3; num++) {

				res[i][j] += mat.m_value[i][num] * this->m_value[num][j];

			}

		}

	}
	this->m_value = res;

	return *this;
}
ge::TransMat& ge::TransMat::SetScaling(double a, double b)
{
	TransMat mat;
	mat.m_value[0][0] = a;
	mat.m_value[1][1] = b;

	vector<vector<double>>res(3, { 0,0,0 });

	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 3; j++) {

			for (int num = 0; num < 3; num++) {

				res[i][j] += mat.m_value[i][num] * this->m_value[num][j];

			}

		}

	}
	this->m_value = res;

	return *this;
}
ge::TransMat& ge::TransMat::SetRotation(double ang)
{
	TransMat mat;
	mat.m_value[0][0] = cos(ang);
	mat.m_value[0][1] = -sin(ang);
	mat.m_value[1][0] = sin(ang);
	mat.m_value[1][1] = cos(ang);

	vector<vector<double>>res(3, { 0,0,0 });
	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 3; j++) {

			for (int num = 0; num < 3; num++) {

				res[i][j] += mat.m_value[i][num] * this->m_value[num][j];

			}

		}

	}
	this->m_value = res;

	return *this;
}
bool ge::TransMat::IsZeros()
{
	for (auto it : this->m_value) {

		for (auto itit : it) {

			if (abs(itit - 0.0) > 0.001)return false;

		}

	}

	return true;
}
void ge::TransMat::Clear()
{
	this->m_value = vector<vector<double>>(3, { 0,0,0 });
	for (int i = 0; i < 3; i++) {

		m_value[i][i] = 1;

	}
}


//---------------------------------------直方图匹配函数定义-----------------------------------------

hs::Histogram::Histogram() {

	this->m_sum = vector<int>(256, 0);
	this->m_p = vector<double>(256, 0.0f);
	this->m_all = 0;

}
hs::Histogram::Histogram(int i) {

	this->m_sum = vector<int>(256, 0);
	this->m_p = vector<double>(256, 1.0 / 256.0);
	this->m_all = 0;

}
void hs::histogramStat(const Mat& inImage, Histogram& hist) {

	hist.m_all = inImage.rows * inImage.cols;

	for (int i = 0; i < inImage.rows; i++) {

		for (int j = 0; j < inImage.cols; j++) {

			hist.m_sum[inImage.at<uchar>(i, j)]++;

		}
	}

	for (int i = 0; i < 256; i++) {

		hist.m_p[i] = (double)hist.m_sum[i] / (double)hist.m_all;

	}


}
void hs::Histogram::Draw(Mat& outImage) {

	outImage = Mat::zeros(300, 256 * 2, CV_8UC1);

	double maxVal = *max_element(this->m_sum.begin(), this->m_sum.end());

	for (int i = 0; i < 256; i++) {

		line(outImage, Point(i * 2, 299), Point(i * 2, 299 - this->m_sum[i] / maxVal * 299.0), Scalar(255, 0, 0), 2);

	}

}
void hs::histogramMatching(const Histogram& src, const Histogram& dst, map<uchar, uchar>& index)
{

	vector<double>Pi(256, 0.0f);
	vector<double>Pj(256, 0.0f);
	Pi[0] = src.m_p[0];
	Pj[0] = dst.m_p[0];

	for (int i = 1; i < 256; i++) {

		Pi[i] = src.m_p[i] + Pi[i - 1];
		Pj[i] = dst.m_p[i] + Pj[i - 1];

	}


	//SML
	//遍历Pi，在Pj中找最接近的
	//找abs(Pj-Pi[itSrc])最小值

	for (int itSrc = 0; itSrc < 256; itSrc++) {

		vector<double>absVec(256, 0.0f);

		for (int i = 0; i < 256; i++) {

			absVec[i] = abs(Pj[i] - Pi[itSrc]);

		}

		uchar close = min_element(absVec.begin(), absVec.end()) - absVec.begin();

		index.insert(make_pair(itSrc, close));

	}



}
void hs::transHistogram(const Mat& inImage, Mat& outImage, map<uchar, uchar>& index) {

	outImage.create(inImage.rows, inImage.cols, inImage.type());

	for (int i = 0; i < inImage.rows; i++) {

		for (int j = 0; j < inImage.cols; j++) {

			outImage.at<uchar>(i, j) = index[inImage.at<uchar>(i, j)];

		}

	}

}


//---------------------------------------交互函数定义-----------------------------------------------

void cw::multipleImage(vector<Mat> imgVector, Mat& dst, int imgCols)
{
	const int MAX_PIXEL = 300;
	int imgNum = imgVector.size();
	//选择图片最大的一边 将最大的边按比例变为300像素
	Size imgOriSize = imgVector[0].size();
	int imgMaxPixel = max(imgOriSize.height, imgOriSize.width);
	//获取最大像素变为MAX_PIXEL的比例因子
	double prop = imgMaxPixel < MAX_PIXEL ? (double)imgMaxPixel / MAX_PIXEL : MAX_PIXEL / (double)imgMaxPixel;
	Size imgStdSize(imgOriSize.width * prop, imgOriSize.height * prop); //窗口显示的标准图像的Size

	Mat imgStd; //标准图片
	Point2i location(0, 0); //坐标点(从0,0开始)
	//构建窗口大小 通道与imageVector[0]的通道一样
	Mat imgWindow(imgStdSize.height * ((imgNum - 1) / imgCols + 1), imgStdSize.width * imgCols, imgVector[0].type());
	for (int i = 0; i < imgNum; i++)
	{
		location.x = (i % imgCols) * imgStdSize.width;
		location.y = (i / imgCols) * imgStdSize.height;
		resize(imgVector[i], imgStd, imgStdSize, prop, prop, INTER_LINEAR); //设置为标准大小
		imgStd.copyTo(imgWindow(Rect(location, imgStdSize)));
	}
	dst = imgWindow;
}
Mat cw::readImage() {

	Mat mat;

	string strPath;

	cout << "请输入图片路径" << endl;
	cin >> strPath;


	mat = imread(strPath, IMREAD_COLOR);
	
	if (mat.empty()) {

		cout << "读取失败" << endl;
		return mat;

	}

	cvtColor(mat, mat, COLOR_RGB2GRAY);
	
	cout << "读取成功" << endl;
	return mat;

}
void cw::saveImage(Mat& mat) {

	string strPath;
	int a;

	cout << "是否要保存图片 【1.保存 0.不保存】" << endl;

	cin >> a;
	if (a == 0) {

		return;
	}
	else if (a == 1) {

		cout << "请输入保存路径" << endl;
		cin >> strPath;

		imwrite(strPath, mat);

		cout << "保存成功" << endl;

	}

}


//---------------------------------------菜单及色彩函数---------------------------------------------

void color(const unsigned short textColor)
{
	if (textColor >= 0 && textColor <= 15)
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), textColor);
	else
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7);
//color(0);   printf("黑色\n");
//color(1);   printf("蓝色\n");
//color(2);   printf("绿色\n");
//color(3);   printf("湖蓝色\n");
//color(4);   printf("红色\n");
//color(5);   printf("紫色\n");
//color(6);   printf("黄色\n");
//color(7);   printf("白色\n");
//color(8);   printf("灰色\n");
//color(9);   printf("淡蓝色\n");
//color(10);  printf("淡绿色\n");
//color(11);  printf("淡浅绿色\n");
//color(12);  printf("淡红色\n");
//color(13);  printf("淡紫色\n");
//color(14);  printf("淡黄色\n");
//color(15);  printf("亮白色\n")
}
int menu()
{
	int menuitem;
	int item;
	while (1)
	{
		menuitem = -1;
		item = -1;
		while (menuitem != 1 && menuitem != 2 && menuitem != 3 && menuitem != 4)
		{
			system("cls");	//清屏 
			color(9);
			printf("                主功能菜单                 \n");
			color(7);

			printf("*******************************************\n");
			printf("**                                       **\n");
			printf("**           1. 灰度变换                 **\n");
			printf("**           2. 局部处理                 **\n");
			printf("**           3. 几何处理                 **\n");
			printf("**           4. 直方图匹配               **\n");
			printf("**           0. 退出                     **\n");
			printf("**                                       **\n");
			printf("*******************************************\n");
			color(9);
			printf("请选择(0-4):");
			color(7);


			int flag = scanf("%d", &menuitem);
			scanf("%*[^\n]"); scanf("%*c");
			while (flag != 1 || menuitem < 0 || menuitem > 4) {
				color(4);
				printf("请输入正确的指令\n");
				color(7);
				flag = scanf("%d", &menuitem);
				scanf("%*[^\n]"); scanf("%*c");
			}

			if (menuitem == 0)
			{
				return END;
			}

		}
		switch (menuitem)
		{
		case 1:
		{
			while (item != 1 && item != 2 && item != 3)
			{

				system("cls");	//清屏 

				color(9);
				printf("                 灰度变换                  \n");
				color(7);

				printf("*******************************************\n");
				printf("**                                       **\n");
				printf("**           1. 线性变换                 **\n");
				printf("**           2. 分段线性变换             **\n");
				printf("**           3. 对数变换                 **\n");
				printf("**           0. 返回                     **\n");
				printf("**                                       **\n");
				printf("*******************************************\n");

				color(9);
				printf("请选择(0-3):");
				color(7);


				int flag = scanf("%d", &item);
				scanf("%*[^\n]"); scanf("%*c");
				while (flag != 1 || item < 0 || item > 4) {
					color(4);
					printf("请输入正确的指令\n");
					color(7);
					flag = scanf("%d", &item);
					scanf("%*[^\n]"); scanf("%*c");
				}


				if (item == 0) break;
			}

			switch (item)
			{
			case 1:
				return GRAY_LINEAR_TRANS;
			case 2:
				return GRAY_PIECEWISE_LINEAR_TRANS;
			case 3:
				return GRAY_LOG_TRANS;
			}
			break;
		}
		case 2:
		{
			while (item != 1 && item != 2 && item != 3 && item != 4)
			{
				system("cls");	//清屏 
				color(9);
				printf("                 局部处理                  \n");

				color(7);
				printf("*******************************************\n");
				printf("**                                       **\n");
				printf("**           1. 高通滤波                 **\n");
				printf("**           2. 低通滤波                 **\n");
				printf("**           3. 中值滤波                 **\n");
				printf("**           4. 自定义滤波器             **\n");
				printf("**           0. 返回                     **\n");
				printf("**                                       **\n");
				printf("*******************************************\n");

				color(9);
				printf("请选择(0-3):");
				color(7);

				int flag = scanf("%d", &item);
				scanf("%*[^\n]"); scanf("%*c");
				while (flag != 1 || item < 0 || item > 4) {
					color(4);
					printf("请输入正确的指令\n");
					color(7);
					flag = scanf("%d", &item);
					scanf("%*[^\n]"); scanf("%*c");
				}				if (item == 0) break;
			}

			switch (item)
			{
			case 1:
				return HIGH_PASS_FILTER;
			case 2:
				return LOW_PASS_FILTER;
			case 3:
				return MID_FILTER;
			case 4:
				return DIY_FILTER;
			}
			break;
		}
		case 3:
		{

			return GEOMETRIC_TRANS;

			//while (item != 1 && item != 2 && item != 3)
			//{
			//	system("cls");	//清屏 
			//	color(9);
			//	printf("                 几何处理                  \n");

			//	color(7);
			//	printf("*******************************************\n");
			//	printf("**                                       **\n");
			//	printf("**           1. 添加平移                 **\n");
			//	printf("**           2. 添加缩放                 **\n");
			//	printf("**           3. 添加旋转                 **\n");
			//	printf("**           0. 返回                     **\n");
			//	printf("**                                       **\n");
			//	printf("*******************************************\n");

			//	color(9);
			//	printf("请选择(0-3):");
			//	color(7);
			//	int flag = scanf("%d", &item);
			//	scanf("%*[^\n]"); scanf("%*c");
			//	while (flag != 1 || item < 0 || item > 3) {
			//		color(4);
			//		printf("请输入正确的指令\n");
			//		color(7);
			//		flag = scanf("%d", &item);
			//		scanf("%*[^\n]"); scanf("%*c");
			//	}				if (item == 0) break;
			//}

			//switch (item)
			//{
			//case 1:
			//	return ADD_TRANSLATION;
			//case 2:
			//	return ADD_SCALE;
			//case 3:
			//	return ADD_ROTATION;
			//}
			//break;

		}
		case 4:
		{
			while (item != 1 && item != 2 && item != 2)
			{
				system("cls");	//清屏 
				color(9);
				printf("                直方图匹配                 \n");

				color(7);
				printf("*******************************************\n");
				printf("**                                       **\n");
				printf("**           1. 直方图均衡化             **\n");
				printf("**           2. 直方图规定化             **\n");
				printf("**           0. 返回                     **\n");
				printf("**                                       **\n");
				printf("*******************************************\n");

				color(9);
				printf("请选择(0-3):");
				color(7);
				int flag = scanf("%d", &item);
				scanf("%*[^\n]"); scanf("%*c");
				while (flag != 1 || item < 0 || item > 2) {
					color(4);
					printf("请输入正确的指令\n");
					color(7);
					flag = scanf("%d", &item);
					scanf("%*[^\n]"); scanf("%*c");
				}				if (item == 0) break;
			}

			switch (item)
			{
			case 1:
				return HISTOGRAM_EQUALITION ;
			case 2:
				return HISTOGRAM_MATCHING;
			}
			break;
		}

		}

	}

}



//***************************************Trackbar辅助结构体及函数***********************************
//**************************************************************************************************

struct TBGeometricTrans {

	Mat m_ori;
	Mat m_cnt;
	Mat m_cntPart;
	ge::TransMat m_transMat;
	int m_posx = 0;
	int m_posy = 0;
	int m_ang = 0;
	int m_scalex = 10;
	int m_scaley = 10;
	int m_max_posx = 1000;
	int m_max_posy = 1000;
	int m_max_ang = 100;
	int m_max_scalex = 20;
	int m_max_scaley = 20;

};
struct TBGrayLinearTrans {

	Mat m_ori;
	Mat m_cnt;
	int a = 1;
	int b = 0;

};

void OnChange(int pos, void* userdata) {

	TBGeometricTrans tb = *((TBGeometricTrans*)userdata);


	tb.m_transMat.Clear();

	tb.m_transMat.SetRotation(3.14 / 50.0 * tb.m_ang).SetTranslation(tb.m_posx, tb.m_posy).SetScaling(tb.m_scalex / 10.0, tb.m_scaley / 10.0);

	ge::geometricTrans(tb.m_ori, tb.m_cnt, tb.m_cntPart, tb.m_transMat);


	imshow("输出窗口1", tb.m_cnt);
	imshow("输出窗口2", tb.m_cntPart);

}
void OnChangeMid(int pos, void* userdata) {

	Mat mat = *((Mat*)userdata);
	Mat outMat;

	sf::fastMidFilter(mat, outMat, pos);
	imshow("输出窗口", outMat);

}




//***************************************调试代码入口***********************************************
//**************************************************************************************************

//#define MYTESTCODE
//-----------
// 
// 运行测试代码时打开此宏，运行正式程序时关闭
// 
//-----------

#ifdef MYTESTCODE

void test() {

	Mat mat, outMat;
	mat = imread("E:\\数图实习\\dip7Q\\dip7Q\\pic\\test.png", IMREAD_COLOR);
	cvtColor(mat, mat, COLOR_RGB2GRAY);
	int n = 3;

	for (int i = 0; i < 15; i++) {

		double start = static_cast<double>(getTickCount());

		sf::fastMidFilter(mat, outMat, n);
		//imshow("inImage", mat);
		//imshow("outImage", outMat);



		double time = ((double)getTickCount() - start) / getTickFrequency();
		
		cout << "n = " << n << "时" << "所用时间为：" << time << "秒" << endl;
		n += 2;

	}

	waitKey(0);
	destroyAllWindows();
	
}
#endif


//***************************************主函数入口*************************************************
//**************************************************************************************************


int main()
{

#ifdef MYTESTCODE

	test();
	return 0;

#endif




	vector<Mat> imageVector;
	Mat dst;


	//几何变换
	TBGeometricTrans temp;

	//灰度变换
	Mat mat, outMat, TMat;
	Vec2i vec2i;
	vector<Vec2i>vec(2, Vec2i());
	int n;


	//直方图规定化
	Mat mat1, mat2, mat3;
	Mat histMat1, histMat2, histMat3;
	hs::Histogram hist1, hist2, hist3;
	map<uchar, uchar>index;
	hs::Histogram hist0(0);

	//空间滤波
    sf::Operator HighPassFilter(sf::HighPass);
    sf::Operator LowpassFilter(sf::Lowpass);
	int size;
	sf::Operator DIYFilter;



	while (1)
	{
		//显示功能菜单,并获得选择的菜单项
		int menuitem = menu();

		switch (menuitem)
		{
		case GRAY_LINEAR_TRANS:

			mat = cw::readImage();
			if (mat.empty())break;
			imshow("inImage",mat);
			waitKey(0);
			cout << "图像灰度值将按照 T(r) = a * r + b 的方式变换" << endl;
			cout << "请输入 a 的值：";
			cin >> vec2i[0];
			cout << "请输入 b 的值：";
			cin >> vec2i[1];
			gt::greyTransformation(mat, outMat, TMat, gt::linearTrans, &vec2i);
			imshow("inImage", mat);
			imshow("outImage", outMat);
			imshow("", TMat);
			waitKey(0);

			cw::saveImage(outMat);
	
			break;
		case GRAY_PIECEWISE_LINEAR_TRANS:

			mat = cw::readImage();
			if (mat.empty())break;
			imshow("inImage", mat);
			waitKey(0);
			cout << "图像灰度值将按照 T(r) = a * r + b 的方式变换" << endl;
			cout << "请输入第一个拐点的 x 值：";
			cin >> vec[0][0];
			cout << "请输入第一个拐点的 y 值：";
			cin >> vec[0][1];
			cout << "请输入第二个拐点的 x 值：";
			cin >> vec[1][0];
			cout << "请输入第er个拐点的 x 值：";
			cin >> vec[1][1];
			gt::greyTransformation(mat, outMat, TMat, gt::piecewiseLinearTrans, &vec);
			imshow("inImage", mat);
			imshow("outImage", outMat);
			imshow("", TMat);
			waitKey(0);

			cw::saveImage(outMat);


			break;
		case GRAY_LOG_TRANS:

			mat = cw::readImage();
			if (mat.empty())break;
			imshow("inImage", mat);
			waitKey(0);
			cout << "图像灰度值将按照 T(r) = c * log ( r + 1 ) 的方式变换" << endl;
			cout << "请输入 c 的值：";
			cin >> n;
			gt::greyTransformation(mat, outMat, TMat, gt::logTrans, &n);
			imshow("inImage", mat);
			imshow("outImage", outMat);
			imshow("", TMat);

			waitKey(0);

			cw::saveImage(outMat);

			break;

		case HIGH_PASS_FILTER:

			mat = cw::readImage();
			if (mat.empty())break;
			imshow("inImage", mat);
			waitKey(0);
			cout << "图像灰度值将与以下滤波器进行滤波处理：" << endl;
			cout << " 0   -1   0" << endl;
			cout << "-1    4  -1" << endl;
			cout << " 0   -1   0" << endl;
			waitKey(0);
			
			sf::linearFilter(mat, outMat, HighPassFilter);
			imshow("inImage", mat);
			imshow("outImage", outMat);
			waitKey(0);

			cw::saveImage(outMat);


			break;
		case LOW_PASS_FILTER:

			mat = cw::readImage();
			if (mat.empty())break;
			imshow("inImage", mat);
			waitKey(0);
			cout << "图像灰度值将与以下滤波器进行滤波处理：" << endl;
			cout << "1/9   1/9   1/9" << endl;
			cout << "1/9   1/9   1/9" << endl;
			cout << "1/9   1/9   1/9" << endl;
			waitKey(0);

			sf::linearFilter(mat, outMat, LowpassFilter);
			imshow("inImage", mat);
			imshow("outImage", outMat);
			waitKey(0);

			cw::saveImage(outMat);

			break;
		case MID_FILTER:


			mat = cw::readImage();
			if (mat.empty())break;
			imshow("inImage", mat);
			waitKey(0);

			size = 3;
			namedWindow("输入窗口", WINDOW_KEEPRATIO);
			createTrackbar("模板大小:", "输入窗口", &size, 150, OnChangeMid, (void*)&mat);

			OnChangeMid(3, (void*)&mat);
			waitKey(0);
			destroyAllWindows();
			break;

		case DIY_FILTER:


			mat = cw::readImage();
			if (mat.empty())break;
			imshow("inImage", mat);
			waitKey(0);


			cout << "请输入滤波器" << endl;
			cin >> DIYFilter.m_value[0] >> DIYFilter.m_value[1] >> DIYFilter.m_value[2] >> DIYFilter.m_value[3] >> DIYFilter.m_value[4] >> DIYFilter.m_value[5] >> DIYFilter.m_value[6] >> DIYFilter.m_value[7] >> DIYFilter.m_value[8];

			DIYFilter.m_size = 3;
			waitKey(0);

			sf::linearFilter(mat, outMat, DIYFilter);
			imshow("inImage", mat);
			imshow("outImage", outMat);
			waitKey(0);

			cw::saveImage(outMat);


			break;

		case GEOMETRIC_TRANS:

			temp.m_ori = cw::readImage();
			if (temp.m_ori.empty())break;
			imshow("inImage", temp.m_ori);
			waitKey(0);

			namedWindow("输入窗口", WINDOW_KEEPRATIO);
			createTrackbar("x:", "输入窗口", &temp.m_posx, temp.m_max_posx, OnChange, (void*)&temp);
			createTrackbar("y:", "输入窗口", &temp.m_posy, temp.m_max_posy, OnChange, (void*)&temp);
			createTrackbar("ang:", "输入窗口", &temp.m_ang, temp.m_max_ang, OnChange, (void*)&temp);
			createTrackbar("scalex:", "输入窗口", &temp.m_scalex, temp.m_max_scalex, OnChange, (void*)&temp);
			createTrackbar("scaley:", "输入窗口", &temp.m_scaley, temp.m_max_scaley, OnChange, (void*)&temp);

			imshow("输入窗口", temp.m_ori);
			OnChange(0, (void*)&temp);
			waitKey(0);

			destroyAllWindows();

			break;
		case HISTOGRAM_EQUALITION:

			cout << "请选择图像" << endl;
			mat1 = cw::readImage();
			if (mat1.empty())break;
			hs::histogramStat(mat1, hist1);
			hist1.Draw(histMat1);
			imshow("mat1", mat1);
			imshow("histMat1", histMat1);
			waitKey(0);
			cw::saveImage(histMat1);


			hs::histogramMatching(hist1, hist0, index);         //计算映射规则
			hs::transHistogram(mat1, mat3, index);              //图像变换
			hs::histogramStat(mat3, hist3);                     //计算新图的直方图
			hist3.Draw(histMat3);                               //绘制新图直方图


			imshow("均衡化后的结果图", mat3);
			waitKey(0);
			cw::saveImage(mat3);

			imshow("均衡化后的直方图", histMat3);
			waitKey(0);
			cw::saveImage(histMat3);


			imageVector.push_back(mat1);
			imageVector.push_back(mat3);
			imageVector.push_back(histMat1);
			imageVector.push_back(histMat3);

			cw::multipleImage(imageVector, dst, 2);
			imshow("multipleWindow", dst);

			waitKey(0);

			destroyAllWindows();
			imageVector.clear();



			index.clear();

			break;
		case HISTOGRAM_MATCHING:

			cout << "请选择原始图像" << endl;
			mat1 = cw::readImage();
			if (mat1.empty())break;
			hs::histogramStat(mat1, hist1);
			hist1.Draw(histMat1);
			imshow("mat1", mat1);
			imshow("histMat1", histMat1);
			waitKey(0);
			cw::saveImage(histMat1);

			cout << "请选择模板图像" << endl;
			mat2 = cw::readImage();
			if (mat2.empty())break;
			hs::histogramStat(mat2, hist2);
			hist2.Draw(histMat2);
			imshow("mat2", mat2);
			imshow("histMat2", histMat2);
			waitKey(0);
			cw::saveImage(histMat2);
			
			
			hs::histogramMatching(hist1, hist2, index);         //计算映射规则
			hs::transHistogram(mat1, mat3, index);              //图像变换
			hs::histogramStat(mat3, hist3);                     //计算新图的直方图
			hist3.Draw(histMat3);                               //绘制新图直方图


			imshow("规定化后的结果图", mat3);
			waitKey(0);
			cw::saveImage(mat3);

			imshow("规定化后的结果图", histMat3);
			waitKey(0);
			cw::saveImage(histMat3);

			imageVector.push_back(mat1);
			imageVector.push_back(mat2);
			imageVector.push_back(mat3);
			imageVector.push_back(histMat1);
			imageVector.push_back(histMat2);
			imageVector.push_back(histMat3);

			cw::multipleImage(imageVector, dst, 3);
			imshow("multipleWindow", dst);

			waitKey(0);

			destroyAllWindows();
			imageVector.clear();
			index.clear();

			break;

		case END:


			waitKey(0);
			destroyAllWindows();
			return 0;

		}
		color(9);
		system("pause");
		color(7);

	}

	
	waitKey(0);
	destroyAllWindows();
	return 0;

}
