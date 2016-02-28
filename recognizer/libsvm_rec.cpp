#define LIB_SVM
#pragma warning(disable:4996)

#ifdef LIB_SVM
#include "svm.h"
struct svm_parameter param;		// SVM�ݒ�p�p�����[�^
struct svm_problem prob;		// �f�[�^�Z�b�g�i���x�����f�[�^�j�E�p�����[�^
struct svm_node *x_space;		// �f�[�^�E�p�����[�^�isvm_problem�̉����ϐ��j
struct svm_model *model;		// �w�K�f�[�^�E�p�����[�^
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
const int CLASS_COUNT = 10;                       // �N���X��
const int VISUAL_WORDS_COUNT = 2000;              // BOW�����x�N�g���̎��� (RGB:1����������)

int recognize(Mat img, string group_id, string dec){


	//�t�@�C�����̐ݒ�
	string group_dictionary = dec + "/libdictionary_" + group_id + ".xml";
	string group_svm = dec + "/libsvm_" + group_id + ".xml";

	// SURFeatureDetector, Extractor�̐ݒ�
	Ptr<FeatureDetector> detector;
	initModule_nonfree();
	detector = FeatureDetector::create(FEATURE_DETECTOR_TYPE);
	detector->set("hessianThreshold", 100);     // 
	detector->set("nOctaves", 4);				// 
	detector->set("nOctaveLayers", 2);          // 
	detector->set("extended", true);            // 

	// BOW�������o��p�����[�^�ݒ�
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;
	extractor = DescriptorExtractor::create(DESCRIPTOR_EXTRACTOR_TYPE);
	matcher = DescriptorMatcher::create("FlannBased");
	int clusterCount = 100;//�N���X�^k�̐�
	TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
	int attempts = 3;
	int flags = KMEANS_PP_CENTERS;

	BOWKMeansTrainer bowtrainer(clusterCount, tc, attempts, flags);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	//�����̓ǂݍ���
	Mat dictionary;
	FileStorage cvfs(group_dictionary, CV_STORAGE_READ);
	FileNode node(cvfs.fs, NULL);
	read(node["dictionary"], dictionary);
	bowDE.setVocabulary(dictionary);

	//�w�K�m�[�h
	svm_node test[100];
	Mat test_bowDescriptor;
	vector<KeyPoint> test_keypoint;
	detector->detect(img, test_keypoint);
	bowDE.compute(img, test_keypoint, test_bowDescriptor);
	for (int i = 0; i < clusterCount; i++){
		test[i].index = i + 1;//index��1~�ł��邽��
		test[i].value = test_bowDescriptor.at<float>(0, i);
	}
	test[100].index = -1;//��؂�p

	// �w�K���鎯�ʊ�̃p�����[�^
	svm_parameter param;
	// SVM�ݒ�l
	param.svm_type = C_SVC;// SVC
	param.kernel_type = RBF;// RBF�i���ˊ��֐��j�J�[�l��
	param.degree = 3;
	param.gamma = 5.0;// �J�[�l���ȂǂŎg����p�����[�^
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1.0;// ���ʈȉ��̌���e�F����\�t�g�}�[�W�����̂��߂̃y�i���e�B�W��C:�������قǃ\�t�g�}�[�W��
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;//�\���̏��𓾂�
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	//���ފ�̃��[�h
	struct svm_model *model;
	model = svm_load_model(group_svm.c_str());

	//����
	double prob_est[model->nr_class];
	float res = svm_predict_probability(model, test, prob_est);
	float res2 = svm_predict(model, test);
	//���ʂ̃\�[�g
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
	// ��n��
	svm_free_and_destroy_model(&model);
	//delete[] test;

	
	return 0;

}

int main(int argc, char *argv[]){

	// �擾�����t�@�C���������ׂĕ\������
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