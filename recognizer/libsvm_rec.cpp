#define LIB_SVM
#pragma warning(disable:4996)

#ifdef LIB_SVM
#include "svm.h"
struct svm_parameter param;		// SVM設定用パラメータ
struct svm_problem prob;		// データセット（ラベル＆データ）・パラメータ
struct svm_node *x_space;		// データ・パラメータ（svm_problemの下部変数）
struct svm_model *model;		// 学習データ・パラメータ
#endif //LIB_SVM

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>


using namespace cv;
using namespace std;
const int DIM = 128;
const int SURF_PARAM = 400;
const string FEATURE_DETECTOR_TYPE = "SURF";      // SIFT, SURF
const string DESCRIPTOR_EXTRACTOR_TYPE = "SURF";  // SIFT, SURF
const int CLASS_COUNT = 10;                       // クラス数
const int VISUAL_WORDS_COUNT = 2000;              // BOW特徴ベクトルの次元 (RGB:1成分あたり)

int recognize(Mat img, string group_id, string dec){


	//ファイル名の設定
	string group_dictionary = dec + "/libdictionary_" + group_id + ".xml";
	string group_svm = dec + "/libsvm_" + group_id + ".xml";

	// SURFeatureDetector, Extractorの設定
	Ptr<FeatureDetector> detector;
	initModule_nonfree();
	detector = FeatureDetector::create(FEATURE_DETECTOR_TYPE);
	detector->set("hessianThreshold", 100);     // 
	detector->set("nOctaves", 4);				// 
	detector->set("nOctaveLayers", 2);          // 
	detector->set("extended", true);            // 

	// BOW特徴抽出器パラメータ設定
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;
	extractor = DescriptorExtractor::create(DESCRIPTOR_EXTRACTOR_TYPE);
	matcher = DescriptorMatcher::create("FlannBased");
	int clusterCount = 100;//クラスタkの数
	TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
	int attempts = 3;
	int flags = KMEANS_PP_CENTERS;

	BOWKMeansTrainer bowtrainer(clusterCount, tc, attempts, flags);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	//辞書の読み込み
	Mat dictionary;
	FileStorage cvfs(group_dictionary, CV_STORAGE_READ);
	FileNode node(cvfs.fs, NULL);
	read(node["dictionary"], dictionary);
	bowDE.setVocabulary(dictionary);

	//学習ノード
	svm_node test[100];
	Mat test_bowDescriptor;
	vector<KeyPoint> test_keypoint;
	detector->detect(img, test_keypoint);
	bowDE.compute(img, test_keypoint, test_bowDescriptor);
	for (int i = 0; i < clusterCount; i++){
		test[i].index = i + 1;//indexは1~であるため
		test[i].value = test_bowDescriptor.at<float>(0, i);
	}
	test[100].index = -1;//区切り用

	// 学習する識別器のパラメータ
	svm_parameter param;
	// SVM設定値
	param.svm_type = C_SVC;// SVC
	param.kernel_type = RBF;// RBF（放射基底関数）カーネル
	param.degree = 3;
	param.gamma = 5.0;// カーネルなどで使われるパラメータ
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1.0;// 一定量以下の誤りを容認するソフトマージン化のためのペナルティ係数C:小さいほどソフトマージン
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;//可能性の情報を得る
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	//分類器のロード
	struct svm_model *model;
	model = svm_load_model(group_svm.c_str());

	//識別
	double prob_est[model->nr_class];
	float res = svm_predict_probability(model, test, prob_est);
	float res2 = svm_predict(model, test);
	//結果のソート
	vector<pair<double, double> > pair(model->nr_class);

	for (int m = 0; m < model->nr_class; m++){
		pair[m] = make_pair(prob_est[m], model->label[m]);
	}

	sort(pair.begin(), pair.end());
	string result;
	for (int m = 0; m < model->nr_class; m++){
		cout << pair[m].second << ",";
	}

	cout << result;
	// 後始末
	svm_free_and_destroy_model(&model);
	//delete[] test;

	
	return 0;

}

int main(int argc, char *argv[]){

	// 取得したファイル名をすべて表示する
	string file_name = argv[1];
	string group_id = argv[2];
	string dec = argv[3];
	Mat img = imread(file_name, 0);
	if (img.empty()) {
		cerr << "Error: Could not open one of the images." << endl;
	}

	float res = recognize(img, group_id, dec);

	return 0;

}