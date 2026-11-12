#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

namespace {

bool hasProvider(
    const std::vector<std::string>& providers,
    const std::string& name
) {
    return std::find(providers.begin(), providers.end(), name) != providers.end();
}

void configureExecutionProvider(
    Ort::SessionOptions& sessionOptions,
    const std::string& requestedDevice
) {
    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    std::string device = requestedDevice;

    if (device == "auto") {
#ifdef __APPLE__
        device = hasProvider(providers, "CoreMLExecutionProvider") ? "metal" : "cpu";
#else
        device = hasProvider(providers, "CUDAExecutionProvider") ? "cuda" : "cpu";
#endif
    }

    if (device == "cuda") {
        if (!hasProvider(providers, "CUDAExecutionProvider")) {
            throw std::runtime_error(
                "CUDA is unavailable in this ONNX Runtime build. "
                "Set ONNXRUNTIME_ROOT to a CUDA-enabled package."
            );
        }

        OrtCUDAProviderOptionsV2* rawOptions = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&rawOptions));
        const auto deleter = [](OrtCUDAProviderOptionsV2* options) {
            Ort::GetApi().ReleaseCUDAProviderOptions(options);
        };
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(deleter)> cudaOptions(
            rawOptions,
            deleter
        );
        sessionOptions.AppendExecutionProvider_CUDA_V2(*cudaOptions);
        std::cout << "Execution provider: CUDA" << std::endl;
        return;
    }

    if (device == "metal") {
        if (!hasProvider(providers, "CoreMLExecutionProvider")) {
            throw std::runtime_error(
                "Metal is unavailable: this ONNX Runtime build has no CoreML provider."
            );
        }
#ifdef __APPLE__
        const uint32_t flags = COREML_FLAG_USE_CPU_AND_GPU;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
            sessionOptions,
            flags
        ));
        std::cout << "Execution provider: Metal (CoreML, CPU+GPU)" << std::endl;
        return;
#else
        throw std::runtime_error("Metal is only supported on Apple platforms.");
#endif
    }

    if (device == "cpu") {
        std::cout << "Execution provider: CPU" << std::endl;
        return;
    }

    throw std::runtime_error(
        "Unknown device: " + requestedDevice + " (expected auto, cpu, cuda or metal)"
    );
}

constexpr int kInputWidth = 256;
constexpr int kInputHeight = 256;
constexpr int kInputChannels = 3;

cv::Mat preprocessFrame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::Mat rgb;
    cv::Mat floatImage;

    cv::resize(
        frame,
        resized,
        cv::Size(kInputWidth, kInputHeight),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    return floatImage;
}

void keepLargestComponent(cv::Mat& alpha) {
    cv::Mat bin;
    cv::threshold(alpha, bin, 0.5, 255.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(
        bin,
        labels,
        stats,
        centroids,
        8
    );
    if (count <= 2) return;

    int best = 1;
    for (int i = 2; i < count; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >
            stats.at<int>(best, cv::CC_STAT_AREA)) {
            best = i;
        }
    }

    cv::Mat keep = (labels == best);
    cv::dilate(
        keep,
        keep,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5})
    );

    cv::Mat drop;
    cv::bitwise_not(keep, drop);
    alpha.setTo(0.0f, drop);
}

void sharpenAlpha(cv::Mat& alpha, float lo = 0.35f, float hi = 0.65f) {
    cv::Mat t;
    alpha.convertTo(t, CV_32F, 1.0 / (hi - lo), -lo / (hi - lo));
    cv::min(t, 1.0, t);
    cv::max(t, 0.0, t);
    cv::Mat t2 = t.mul(t);
    cv::Mat poly = 3.0 - 2.0 * t;
    alpha = t2.mul(poly);
}

cv::Mat extractMask(Ort::Value& output) {
    const auto info = output.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();

    if (shape.size() != 4 ||
        shape[0] != 1 ||
        shape[3] != 1 ||
        shape[1] <= 0 ||
        shape[2] <= 0) {
        std::string shapeText = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) {
                shapeText += ", ";
            }
            shapeText += std::to_string(shape[i]);
        }
        shapeText += "]";

        throw std::runtime_error(
            "Unexpected segmentation output shape: " + shapeText
        );
    }

    const int height = static_cast<int>(shape[1]);
    const int width = static_cast<int>(shape[2]);
    return cv::Mat(
        height,
        width,
        CV_32F,
        const_cast<float*>(output.GetTensorData<float>())
    ).clone();
}

cv::Mat prepareBackground(
    const cv::Mat& backgroundSource,
    const cv::Size& frameSize,
    const int frameType
) {
    if (backgroundSource.empty()) {
        return cv::Mat(frameSize, frameType, cv::Scalar(0, 255, 0));
    }

    cv::Mat background;
    cv::resize(
        backgroundSource,
        background,
        frameSize,
        0.0,
        0.0,
        cv::INTER_LINEAR
    );
    return background;
}

class AsyncSegmenter {
public:
    AsyncSegmenter(const std::string& modelPath, const std::string& device)
        : environment(ORT_LOGGING_LEVEL_WARNING, "segmentation"),
          sessionOptions(),
          session(nullptr) {
        configureExecutionProvider(sessionOptions, device);
        session = Ort::Session(environment, modelPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        auto allocatedInputName = session.GetInputNameAllocated(0, allocator);
        auto allocatedOutputName = session.GetOutputNameAllocated(0, allocator);
        inputName = allocatedInputName.get();
        outputName = allocatedOutputName.get();
        inputNames = {inputName.c_str()};
        outputNames = {outputName.c_str()};
        worker = std::thread(&AsyncSegmenter::run, this);
    }

    ~AsyncSegmenter() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        inputReady.notify_one();
        outputReady.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    AsyncSegmenter(const AsyncSegmenter&) = delete;
    AsyncSegmenter& operator=(const AsyncSegmenter&) = delete;

    void submit(const cv::Mat& frame) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            pendingFrame = frame.clone();
            hasPendingFrame = true;
        }
        inputReady.notify_one();
    }

    bool getLatestAlpha(cv::Mat& destination, uint64_t& revision) {
        std::lock_guard<std::mutex> lock(mutex);
        if (latestAlpha.empty() || revision == alphaRevision) {
            return false;
        }

        destination = latestAlpha;
        revision = alphaRevision;
        return true;
    }

    void waitForNextAlpha(cv::Mat& destination, uint64_t& revision) {
        std::unique_lock<std::mutex> lock(mutex);
        outputReady.wait(lock, [this, &revision] {
            return stopping || workerError || alphaRevision > revision;
        });
        if (workerError) {
            std::rethrow_exception(workerError);
        }
        if (stopping) {
            throw std::runtime_error("Segmentation stopped");
        }

        destination = latestAlpha;
        revision = alphaRevision;
    }

    void throwIfFailed() const {
        std::lock_guard<std::mutex> lock(mutex);
        if (workerError) {
            std::rethrow_exception(workerError);
        }
    }

private:
    void run() {
        try {
            const Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );
            const std::array<int64_t, 4> inputShape = {
                1,
                kInputHeight,
                kInputWidth,
                kInputChannels
            };
            cv::Mat previousAlpha;

            while (true) {
                cv::Mat frame;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    inputReady.wait(lock, [this] {
                        return stopping || hasPendingFrame;
                    });
                    if (stopping) return;

                    frame = std::move(pendingFrame);
                    pendingFrame.release();
                    hasPendingFrame = false;
                }

                cv::Mat inputImage = preprocessFrame(frame);
                Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    inputImage.ptr<float>(),
                    inputImage.total() * inputImage.channels(),
                    inputShape.data(),
                    inputShape.size()
                );

                std::vector<Ort::Value> outputs = session.Run(
                    Ort::RunOptions{nullptr},
                    inputNames.data(),
                    &inputTensor,
                    1,
                    outputNames.data(),
                    1
                );

                cv::Mat alpha = extractMask(outputs[0]);
                if (previousAlpha.empty() || previousAlpha.size() != alpha.size()) {
                    previousAlpha = alpha.clone();
                } else {
                    cv::addWeighted(
                        alpha,
                        0.6,
                        previousAlpha,
                        0.4,
                        0.0,
                        previousAlpha
                    );
                }
                alpha = previousAlpha.clone();
                keepLargestComponent(alpha);

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    latestAlpha = std::move(alpha);
                    ++alphaRevision;
                }
                outputReady.notify_all();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                workerError = std::current_exception();
            }
            outputReady.notify_all();
        }
    }

    Ort::Env environment;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    std::string inputName;
    std::string outputName;
    std::array<const char*, 1> inputNames{};
    std::array<const char*, 1> outputNames{};

    mutable std::mutex mutex;
    std::condition_variable inputReady;
    std::condition_variable outputReady;
    cv::Mat pendingFrame;
    cv::Mat latestAlpha;
    bool hasPendingFrame = false;
    bool stopping = false;
    uint64_t alphaRevision = 0;
    std::exception_ptr workerError;
    std::thread worker;
};

void printUsage(const char* executable) {
    std::cout
        << "Использование:\n"
        << "  " << executable
        << " [--camera N | --video FILE] [--bg IMAGE] [--model FILE] [--out FILE]"
        << " [--device auto|cpu|cuda|metal]\n"
        << "  " << executable << " [video_file] [background_image]\n\n"
        << "Без аргументов используются камера 1 и зелёный фон.\n"
        << "--device выбирает ускоритель; metal использует CoreML на Apple GPU.\n"
        << "--out сохраняет результат без открытия окон.\n"
        << "Для выхода из оконного режима нажмите Q или Esc." << std::endl;
}

}

int main(int argc, char** argv) {
    try {
        std::string modelPath = "../model.onnx";
        std::string videoPath;
        std::string backgroundPath;
        std::string outputPath;
        std::string device = "auto";
        int cameraIndex = 1;
        std::vector<std::string> positional;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                printUsage(argv[0]);
                return 0;
            }

            auto nextValue = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value after " + argument);
                }
                return argv[++i];
            };

            if (argument == "--video") {
                videoPath = nextValue();
            } else if (argument == "--bg") {
                backgroundPath = nextValue();
            } else if (argument == "--model") {
                modelPath = nextValue();
            } else if (argument == "--out") {
                outputPath = nextValue();
            } else if (argument == "--device") {
                device = nextValue();
            } else if (argument == "--camera") {
                const std::string value = nextValue();
                size_t parsed = 0;
                cameraIndex = std::stoi(value, &parsed);
                if (parsed != value.size() || cameraIndex < 0) {
                    throw std::runtime_error("Invalid camera index: " + value);
                }
            } else if (!argument.empty() && argument[0] == '-') {
                throw std::runtime_error("Unknown option: " + argument);
            } else {
                positional.push_back(argument);
            }
        }

        if (positional.size() > 2) {
            printUsage(argv[0]);
            return 1;
        }
        if (!positional.empty()) {
            if (videoPath.empty()) {
                videoPath = positional[0];
            } else if (backgroundPath.empty()) {
                backgroundPath = positional[0];
            } else {
                throw std::runtime_error("Too many input arguments");
            }
        }
        if (positional.size() == 2) {
            if (!backgroundPath.empty()) {
                throw std::runtime_error("Background image specified twice");
            }
            backgroundPath = positional[1];
        }

        const bool useVideoFile = !videoPath.empty();
        const bool saveOutput = !outputPath.empty();
        cv::VideoCapture capture;
        if (useVideoFile) {
            capture.open(videoPath);
        } else {
            capture.open(cameraIndex);
        }

        if (!capture.isOpened()) {
            std::cerr << "Cannot open "
                      << (useVideoFile
                          ? "video: " + videoPath
                          : "camera " + std::to_string(cameraIndex))
                      << std::endl;
            return 1;
        }

        cv::Mat backgroundSource;
        if (!backgroundPath.empty()) {
            backgroundSource = cv::imread(backgroundPath);
            if (backgroundSource.empty()) {
                std::cerr << "Cannot load background image: "
                          << backgroundPath << std::endl;
                return 1;
            }
        }

        AsyncSegmenter segmenter(modelPath, device);
        cv::VideoWriter outputWriter;

        cv::Mat frame;
        cv::Mat alpha3;
        cv::Mat background;
        cv::Size cachedBackgroundSize;
        int cachedBackgroundType = -1;
        uint64_t alphaRevision = 0;
        double smoothedFps = 0.0;
        const double sourceFps = capture.get(cv::CAP_PROP_FPS);

        while (true) {
            const int64_t frameStart = cv::getTickCount();
            if (!capture.read(frame) || frame.empty()) {
                if (useVideoFile) {
                    std::cout << "Video finished" << std::endl;
                } else {
                    std::cerr << "Cannot read frame from camera" << std::endl;
                }
                break;
            }

            segmenter.submit(frame);
            cv::Mat alpha;
            if (saveOutput) {
                segmenter.waitForNextAlpha(alpha, alphaRevision);
            } else {
                segmenter.throwIfFailed();
                segmenter.getLatestAlpha(alpha, alphaRevision);
            }

            if (!alpha.empty()) {
                cv::resize(
                    alpha,
                    alpha,
                    frame.size(),
                    0.0,
                    0.0,
                    cv::INTER_CUBIC
                );
                cv::min(alpha, 1.0, alpha);
                cv::max(alpha, 0.0, alpha);
                cv::GaussianBlur(alpha, alpha, cv::Size(7, 7), 0.0);
                sharpenAlpha(alpha);
                cv::cvtColor(alpha, alpha3, cv::COLOR_GRAY2BGR);
            }

            cv::Mat result;
            if (alpha3.empty()) {
                result = frame.clone();
            } else {
                if (background.empty() ||
                    cachedBackgroundSize != frame.size() ||
                    cachedBackgroundType != frame.type()) {
                    background = prepareBackground(
                        backgroundSource,
                        frame.size(),
                        frame.type()
                    );
                    cachedBackgroundSize = frame.size();
                    cachedBackgroundType = frame.type();
                }

                cv::Mat frameFloat;
                cv::Mat backgroundFloat;
                frame.convertTo(frameFloat, CV_32FC3);
                background.convertTo(backgroundFloat, CV_32FC3);

                const cv::Mat inverseAlpha = cv::Scalar::all(1.0) - alpha3;
                const cv::Mat resultFloat =
                    frameFloat.mul(alpha3) + backgroundFloat.mul(inverseAlpha);
                resultFloat.convertTo(result, CV_8UC3);
            }

            if (saveOutput) {
                if (!outputWriter.isOpened()) {
                    const double outputFps = sourceFps > 0.0 ? sourceFps : 30.0;
                    outputWriter.open(
                        outputPath,
                        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                        outputFps,
                        frame.size()
                    );
                    if (!outputWriter.isOpened()) {
                        throw std::runtime_error(
                            "Cannot open output video: " + outputPath
                        );
                    }
                }
                outputWriter.write(result);
                continue;
            }

            const double elapsed =
                static_cast<double>(cv::getTickCount() - frameStart) /
                cv::getTickFrequency();
            if (elapsed > 0.0) {
                const double currentFps = 1.0 / elapsed;
                smoothedFps = smoothedFps == 0.0
                    ? currentFps
                    : 0.9 * smoothedFps + 0.1 * currentFps;
            }

            cv::putText(
                result,
                "FPS: " + std::to_string(static_cast<int>(smoothedFps)),
                cv::Point(15, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2
            );

            cv::imshow("Original", frame);
            cv::imshow("Background replacement", result);

            int delay = 1;
            if (sourceFps > 0.0) {
                const double frameIntervalMs = 1000.0 / sourceFps;
                delay = std::max(1, static_cast<int>(frameIntervalMs - elapsed * 1000.0));
            }
            const int key = cv::waitKey(delay);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }
        }

        capture.release();
        const bool outputSaved = outputWriter.isOpened();
        outputWriter.release();
        if (!saveOutput) {
            cv::destroyAllWindows();
        } else if (outputSaved) {
            std::cout << "Output saved: " << outputPath << std::endl;
        }
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 1;
}
