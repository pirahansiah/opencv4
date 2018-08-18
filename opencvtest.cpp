#include "stdafx.h"
#include "opencvtest.h"
cv::String path = "C:/opencv4/OpenCV4VS2017/DeepLearningFiles/";
/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const cv::Mat &probBlob, int *classId, double *classProb)
{
	cv::Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	cv::Point classNumber;
	cv::minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}
static std::vector<cv::String> readClassNames(const char *filename = "C:/opencv4/OpenCV4VS2017/DeepLearningFiles/synset_words.txt")
{
	std::vector<cv::String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}

static std::vector<cv::String> readClassNamesTensorFlow(const char *filename = "C:/opencv4/OpenCV4VS2017/DeepLearningFiles/imagenet_comp_graph_label_strings.txt")
{
	std::vector<cv::String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}
opencvtest::opencvtest()
{
	//cv::String filePath = "C:/opencv4/OpenCV4VS2017/OpenCV4VS2017/opencv-logo.png";
	//cv::Mat src = cv::imread(filePath);
	//cv::imshow("test", src);
	//cv::waitKey(10000);
}


opencvtest::~opencvtest()
{
}

void opencvtest::deepLearningCaffeModel()
{	
	float scale = 1.0;
	cv::Scalar meanG = cv::Scalar(104, 117, 123);
	bool swapRB = true;
	int inpWidth = 224;
	int inpHeight = 224;	
	cv::String model = path + "bvlc_googlenet.caffemodel";
	cv::String config = path + "bvlc_googlenet.prototxt";
	cv::String nameClass = path + "synset_words.txt";
	cv::String nameImage = path + "space_shuttle.jpg";
	cv::String framework;
	int backendId = 0;
	int targetId = 0;
	cv::dnn::Net net;
	cv::Mat blob;
	cv::Mat img = imread(nameImage);
	net = cv::dnn::readNet(model, config);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
	cv::dnn::blobFromImage(img, blob, scale, cv::Size(inpWidth, inpHeight), meanG, swapRB, false);
	net.setInput(blob);
	cv::Mat prob = net.forward();
	cv::Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	std::vector<double> layersTimes;
	double freq = cv::getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = cv::format("Inference time: %.2f ms", t);
	putText(img, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
	cv::imshow("https://www.youtube.com/tiziran", img);
	std::vector<cv::String> classNames = readClassNames();	
	std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
	std::cout << "Probability: " << confidence * 100 << "%" << std::endl;
	cv::waitKey(10000);
}

void opencvtest::deepLearningTensorFlowModel()
{	//https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip 
	float scale = 1.0;
	cv::Scalar meanG = cv::Scalar(104, 117, 123);
	bool swapRB = true;
	int inpWidth = 224;
	int inpHeight = 224;
	cv::String model = path + "tensorflow_inception_graph.pb";
	cv::String nameImage = path + "space_shuttle.jpg";
	cv::String framework;
	int backendId = 0;
	int targetId = 0;
	cv::dnn::Net net;
	cv::Mat blob;
	cv::Mat img = imread(nameImage);	
	net = cv::dnn::readNetFromTensorflow(model);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
	cv::dnn::blobFromImage(img, blob, scale, cv::Size(inpWidth, inpHeight), meanG, swapRB, false);
	net.setInput(blob);
	cv::Mat prob = net.forward();
	cv::Point classIdPoint;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	std::vector<double> layersTimes;
	double freq = cv::getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = cv::format("Inference time: %.2f ms", t);
	putText(img, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
	cv::imshow("https://www.youtube.com/tiziran", img);
	std::vector<cv::String> classNames = readClassNamesTensorFlow();
	std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
	std::cout << "Probability: " << confidence * 100 << "%" << std::endl;
	cv::waitKey(10000);

}
