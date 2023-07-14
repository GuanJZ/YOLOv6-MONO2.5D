#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.65
#define BBOX_CONF_THRESH 0.3

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
const int num_class = 4;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "outputs";

static const char* class_names[] = {
        "pedestrian", "cyclist", "car", "big_vehicle"
};


static Logger gLogger;


cv::Mat static_resize(cv::Mat& img) {
    // resize
    auto start = std::chrono::system_clock::now();
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    auto end = std::chrono::system_clock::now();
    std::cout << "delay of preprocess for resize: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    // pad
    start = std::chrono::system_clock::now();
    cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
//    re.copyTo(out(cv::Rect(out.cols - re.cols, (out.rows - re.rows) / 2, re.cols, re.rows)));
    end = std::chrono::system_clock::now();
    std::cout << "delay of preprocess for pad: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;
    return out;
}

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    float W, H, L;
    float Ry;
    float offset_x, offset_y;
};

struct InterObject
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    float W_log, H_log, L_log;
    float cos1, sin1, cos2, sin2;
    float conf_orient1, conf_orient2;
    float offset_x, offset_y;
};


static inline float intersection_area(const InterObject& a, const InterObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<InterObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<InterObject>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<InterObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const InterObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const InterObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void generate_yolo_proposals(float* feat_blob, int output_size, float prob_threshold, std::vector<InterObject>& objects)
{
    auto dets = output_size / (num_class + 5 + 3+4+2+2);
    for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
    {
        const int basic_pos = boxs_idx *(num_class + 5 + 3+4+2+2);
        float x_center = feat_blob[basic_pos+0];
        float y_center = feat_blob[basic_pos+1];
        float w = feat_blob[basic_pos+2];
        float h = feat_blob[basic_pos+3];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float box_objectness = feat_blob[basic_pos+4];
        float H_log = feat_blob[basic_pos+4+num_class+1];
        float W_log = feat_blob[basic_pos+4+num_class+2];
        float L_log = feat_blob[basic_pos+4+num_class+3];
        float cos1 = feat_blob[basic_pos+4+num_class+4];
        float sin1 = feat_blob[basic_pos+4+num_class+5];
        float cos2 = feat_blob[basic_pos+4+num_class+6];
        float sin2 = feat_blob[basic_pos+4+num_class+7];
        float conf_orient1 = feat_blob[basic_pos+4+num_class+8];
        float conf_orient2 = feat_blob[basic_pos+4+num_class+9];
        float offset_x = feat_blob[basic_pos+4+num_class+10];
        float offset_y = feat_blob[basic_pos+4+num_class+11];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                InterObject obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;
                obj.H_log = H_log;
                obj.L_log = L_log;
                obj.W_log = W_log;
                obj.cos1 = cos1;
                obj.sin1 = sin1;
                obj.cos2 = cos2;
                obj.sin2 = sin2;
                obj.conf_orient1 = conf_orient1;
                obj.conf_orient2 =conf_orient2;
                obj.offset_x = offset_x;
                obj.offset_y = offset_y;

                objects.push_back(obj);
            }

        } // class loop
    }

}

float* blobFromImage(cv::Mat& img){
    auto start = std::chrono::system_clock::now();
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    auto end = std::chrono::system_clock::now();
    std::cout << "delay of preprocess for COLOR_BGR2RGB: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    start = std::chrono::system_clock::now();
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
//    std::ofstream f1("yolov6_preprocess_cpp.txt");
//    for(size_t i=0; i < channels*img_h*img_w; i++)
//    {
//        f1 << blob[i] << " ";
//    }
    end = std::chrono::system_clock::now();
    std::cout << "delay of preprocess for hwc2chw: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;
    return blob;
}


static void decode_outputs(float* prob, int output_size, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        auto start = std::chrono::system_clock::now();
        std::vector<InterObject> proposals;
        generate_yolo_proposals(prob, output_size, BBOX_CONF_THRESH, proposals);
        auto end1 = std::chrono::system_clock::now();
        std::cout << "delay of generate_yolo_proposals: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start).count() / 1000.0 << "ms" << std::endl;
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);
        auto end2 = std::chrono::system_clock::now();
        std::cout << "delay of qsort_descent_inplace: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count() / 1000.0 << "ms" << std::endl;

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);
        auto end3 = std::chrono::system_clock::now();
        std::cout << "delay of nms_sorted_bboxes: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count() / 1000.0 << "ms" << std::endl;

        int count = picked.size();
        std::cout << "num of boxes after nms: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
//            objects[i] = proposals[picked[i]];
            // adjust offset to original unpadded
            float x0 = (proposals[picked[i]].rect.x) / scale;
            float y0 = (proposals[picked[i]].rect.y) / scale;
            float x1 = (proposals[picked[i]].rect.x + proposals[picked[i]].rect.width) / scale;
            float y1 = (proposals[picked[i]].rect.y + proposals[picked[i]].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
            objects[i].label = proposals[picked[i]].label;
            objects[i].prob = proposals[picked[i]].prob;
            objects[i].H = exp(proposals[picked[i]].H_log);
            objects[i].W = exp(proposals[picked[i]].W_log);
            objects[i].L = exp(proposals[picked[i]].L_log);

            // 计算theta
            float intrinsic_fx = 2183.375019;
            float fovx = 2 * atan2(img_w, 2*intrinsic_fx);
            float center = (x0 + y0) / 2.0;
            float dx = center - (img_w / 2);
            float mult = (dx>0) ? 1:-1;
            float theta = mult * atan((2 * mult * dx * tan(fovx / 2)) / img_w);

            // 解码alpha
            float alpha_decode;
            if (proposals[picked[i]].conf_orient1 > proposals[picked[i]].conf_orient2)
            {
                alpha_decode = atan2(proposals[picked[i]].sin1, proposals[picked[i]].cos1) + (0 + 0.5 -1)*3.1415926;
            } else{
                alpha_decode = atan2(proposals[picked[i]].sin2, proposals[picked[i]].cos2) + (1 + 0.5 -1)*3.1415926;
            }
            objects[i].Ry = theta + alpha_decode;
            objects[i].offset_y = std::max(std::min(proposals[picked[i]].offset_y / scale, (float)(img_h - 1)), 0.f);
            objects[i].offset_x = std::max(std::min(proposals[picked[i]].offset_x / scale, (float)(img_w - 1)), 0.f);

        }
        auto end4 = std::chrono::system_clock::now();
        std::cout << "delay of decode output: ";
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end4 - end3).count() / 1000.0 << "ms" << std::endl;
}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
{

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%% %0.1f %0.1f", class_names[obj.label], obj.prob * 100, obj.rect.x, obj.rect.y);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    cv::imwrite("det_res.jpg", image);
    fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
}


void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < 5; i++){
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(1, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    auto start = std::chrono::system_clock::now();
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference delay: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;


    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
            std::cout << "load file" << std::endl;
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov6 ../model_trt.engine -i ../*.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }
    const std::string input_image_path {argv[3]};

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    std::cout << "runtime ptr create" << std::endl;
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    std::cout << "engine ptr create" << std::endl;
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    std::cout << "context ptr create" << std::endl;

    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    std::cout << "output size: " << output_size << std::endl;
    static float* prob = new float[output_size];

    cv::Mat img = cv::imread(input_image_path);

    auto start = std::chrono::system_clock::now();
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);
    float* blob;
    blob = blobFromImage(pr_img);
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    auto end = std::chrono::system_clock::now();
    std::cout << "delay of whole preprocess: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    // run inference
    doInference(*context, blob, prob, output_size, pr_img.size());

    start = std::chrono::system_clock::now();
    std::vector<Object> objects;
    decode_outputs(prob, output_size, objects, scale, img_w, img_h);
    end = std::chrono::system_clock::now();
    std::cout << "delay of whole postprocess: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    draw_objects(img, objects, input_image_path);
    auto end1 = std::chrono::system_clock::now();
    std::cout << "delay of draw_objects: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end1 - end).count() / 1000.0 << "ms" << std::endl;
    // delete the pointer to the float
    delete blob;
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}