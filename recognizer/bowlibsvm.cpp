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
#include <math.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <exception>

using namespace cv;
using namespace std;
const int DIM = 128;
const int SURF_PARAM = 400;
const int MAX_CLUSTER = 25;//�N���X�^k�̐�
const string FEATURE_DETECTOR_TYPE = "SURF";      // SIFT, SURF
const string DESCRIPTOR_EXTRACTOR_TYPE = "SURF";  // SIFT, SURF
const int CLASS_COUNT = 10;                       // �N���X��
const int VISUAL_WORDS_COUNT = 2000;              // BOW�����x�N�g���̎��� (RGB:1����������)


int recognize(vector<string> file_list, vector<int> class_num, string group_id, string dec){

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

	// data�̊e�摜����Ǐ������ʂ𒊏o
	//�����x�N�g�������o���A�w�K�pDB�����
	for (int n = 0; n < (int)file_list.size(); n++) {
		Mat img = imread(file_list[n], 0);
		if (img.empty()) {
			cerr << "Error: Could not open one of the images." << endl;
		}
		vector<KeyPoint> keypoint;
		Mat features;
		detector->detect(img, keypoint);
		extractor->compute(img, keypoint, features);
		bowtrainer.add(features);//�����ɒǉ����Ă����DBOWKMeansTrainer�^�̕ϐ�

	}
	//�����̍쐬
	Mat dictionary = bowtrainer.cluster();
	bowDE.setVocabulary(dictionary);
	//�����̃t�@�C���ւ̕ۑ�
	FileStorage cvfs2(group_dictionary, CV_STORAGE_WRITE);
	write(cvfs2, "dictionary", dictionary);
	cvfs2.release();
	//SVM�w�K�f�[�^
#ifdef LIB_SVM
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

	//�w�K�f�[�^
	svm_problem prob;

	// �f�[�^�Z�b�g�̃p�����[�^�ݒ�
	prob.l = file_list.size();	// �f�[�^�Z�b�g��

	// �e�p�����[�^�̃������̈�m��
	prob.y = new double[prob.l];		// ���x��
	prob.x = new svm_node *[prob.l];		// �f�[�^�Z�b�g�̕������f�[�^���[��Ԃ��쐬
	x_space = new svm_node[(clusterCount + 1)*prob.l];

	// �f�[�^�Z�b�g�E�p�����[�^�ւ̐��l����
	for (int m = 0; m < prob.l; m++){
		Mat test_img = imread(file_list[m], 0);
		if (test_img.empty()) {
			cerr << "Error: Could not open one of the images." << endl;
		}
		Mat test_bowDescriptor;
		vector<KeyPoint> test_keypoint;
		detector->detect(test_img, test_keypoint);
		bowDE.compute(test_img, test_keypoint, test_bowDescriptor);

		prob.y[m] = class_num[m];//���x��

		for (int j = 0; j<clusterCount; j++){
			x_space[(clusterCount + 1)*m + j].index = j + 1;			// �f�[�^�ԍ��̓���
			x_space[(clusterCount + 1)*m + j].value = test_bowDescriptor.at<float>(0, j);	// �f�[�^�l�̓��� 
		}
		x_space[(clusterCount + 1)*m + clusterCount].index = -1;
		prob.x[m] = &x_space[(clusterCount + 1)*m];	// prob.x��x_space�Ƃ̊֌W�t
	}

	//�w�K
	cout << "Ready to train ..." << endl;
	svm_model *model = svm_train( &prob, &param );
	cout << "Finished ..." << endl;

	//���ފ�̕ۑ�
	svm_save_model(group_svm.c_str(), model);

	//����

	// ��n��
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
	// �������̈�̊J��
	delete[] prob.y;
	delete[] prob.x;
	delete[] x_space;

#else//libsvm���g�p
	Mat flagmat(file_list.size(), 1, CV_32SC1);
	Mat datamat = Mat(file_list.size(), clusterCount, CV_32FC1) * 0;

	for (int m = 0; m < (int)file_list.size(); m++) {
		Mat test_img = imread(file_list[m], 0);
		if (test_img.empty()) {
			cerr << "Error: Could not open one of the images." << endl;
		}
		Mat test_bowDescriptor;
		vector<KeyPoint> test_keypoint;
		detector->detect(test_img, test_keypoint);
		bowDE.compute(test_img, test_keypoint, test_bowDescriptor);
		for (int i = 0; i < clusterCount; i++){
			datamat.at<float>(m, i) = test_bowDescriptor.at<float>(0, i);
		}
		for (int p = 0; p < (int)class_list.size(); p++){
			int loc = file_list[m].find(class_list[p], 0);
			if (loc != string::npos)
				flagmat.at<int>(m, 1) = p;
		}
	}
	//svm�̐ݒ�
	SVM svm;
	SVMParams svmParams;
	// SVM�̃p�����[�^��ݒ�
	svmParams.svm_type = SVM::C_SVC;    // C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR
	svmParams.kernel_type = SVM::RBF;   // LINEAR, POLY, RBF, SIGMOID
	svmParams.C = 1.0;	//���ʈȉ��̌���e�F����\�t�g�}�[�W�����̂��߂̃y�i���e�B�W��C
	svmParams.gamma = 5.0;
	//svmParams.degree = 10.0;
	//���ދ@�̍쐬�ƕۑ�
	svm.train(datamat, flagmat, Mat(), Mat(), svmParams);
	svm.save("/home/sit-user-15/recognizertest/svm_image_test.xml");
#endif//LIB_SVM

	return 0;

}

int main(int argc, char *argv[]){
	string group_id = argv[1];
	string dec = argv[2];

	vector<int> class_num;
	int n = 0;
	string file_name;
	vector<string> file_list;
	for (int m = 0; m < argc; m++){
		cout << m << ":" << argv[m] << endl;
	}

	// \data�̒��̃t�@�C������file_list�Ɋi�[����
	for (int i = 3; i < argc; i++){
		file_name = argv[i];
		file_list.push_back(file_name);
		i++;
		n = atoi(argv[i]);
		class_num.push_back(n);;//class_numt��class�����i�[
		cout << file_name << endl;
		cout << i << endl;
		cout << argc << endl;
	}

	if (class_num.size() > 1){
		cout << "making recognizer" << endl;
		recognize(file_list, class_num, group_id, dec);
		cout << "finish" << endl;
	}
	return 0;

}