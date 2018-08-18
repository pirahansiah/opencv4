#pragma once

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect.hpp"
class opencvtest
{
public:
	opencvtest();	
	~opencvtest();

	void deepLearningCaffeModel();
	void deepLearningTensorFlowModel();
};

