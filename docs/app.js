const IMAGE_SIZE = 256;
const CLASS_NAMES = ["Cat", "Dog"];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const modelStatus = document.getElementById("modelStatus");
const imageInput = document.getElementById("imageInput");
const uploadButton = document.getElementById("uploadButton");
const dropZone = document.getElementById("dropZone");
const previewCanvas = document.getElementById("previewCanvas");
const previewCaption = document.getElementById("previewCaption");
const predictionLabel = document.getElementById("predictionLabel");
const catBar = document.getElementById("catBar");
const dogBar = document.getElementById("dogBar");
const catValue = document.getElementById("catValue");
const dogValue = document.getElementById("dogValue");

let session = null;
let pendingFile = null;

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    session = await ort.InferenceSession.create("./model.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    modelStatus.textContent = "모델 준비 완료";
    modelStatus.classList.add("ready");
    if (pendingFile) {
      const file = pendingFile;
      pendingFile = null;
      classifyImage(file);
    }
  } catch (error) {
    modelStatus.textContent = "모델 로드 실패";
    console.error(error);
  }
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function drawImageToCanvas(image) {
  const context = previewCanvas.getContext("2d", { willReadFrequently: true });
  context.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  context.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  return context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
}

function imageDataToTensor(imageData) {
  const pixels = imageData.data;
  const channelSize = IMAGE_SIZE * IMAGE_SIZE;
  const input = new Float32Array(3 * channelSize);

  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      const pixelIndex = y * IMAGE_SIZE + x;
      const rgbaIndex = pixelIndex * 4;

      const red = pixels[rgbaIndex] / 255;
      const green = pixels[rgbaIndex + 1] / 255;
      const blue = pixels[rgbaIndex + 2] / 255;

      input[pixelIndex] = (red - MEAN[0]) / STD[0];
      input[channelSize + pixelIndex] = (green - MEAN[1]) / STD[1];
      input[channelSize * 2 + pixelIndex] = (blue - MEAN[2]) / STD[2];
    }
  }

  return new ort.Tensor("float32", input, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function updateResult(probabilities) {
  const catPercent = probabilities[0] * 100;
  const dogPercent = probabilities[1] * 100;
  const predictedIndex = catPercent >= dogPercent ? 0 : 1;
  const predictedPercent = probabilities[predictedIndex] * 100;

  predictionLabel.textContent = `${CLASS_NAMES[predictedIndex]} ${predictedPercent.toFixed(1)}%`;
  catBar.style.width = `${catPercent.toFixed(1)}%`;
  dogBar.style.width = `${dogPercent.toFixed(1)}%`;
  catValue.textContent = `${catPercent.toFixed(1)}%`;
  dogValue.textContent = `${dogPercent.toFixed(1)}%`;
}

async function classifyImage(file) {
  const imageUrl = URL.createObjectURL(file);
  const imageName = file.name || "붙여넣은 이미지";
  const image = new Image();
  image.onload = async () => {
    try {
      const imageData = drawImageToCanvas(image);
      previewCaption.textContent = `${imageName} -> 256 x 256 입력`;

      if (!session) {
        pendingFile = file;
        predictionLabel.textContent = "모델 준비 중";
        return;
      }

      const inputTensor = imageDataToTensor(imageData);
      const feeds = { input: inputTensor };
      const results = await session.run(feeds);
      const logits = Array.from(results.logits.data);
      const probabilities = softmax(logits);

      updateResult(probabilities);
    } catch (error) {
      predictionLabel.textContent = "분류 실패";
      console.error(error);
    } finally {
      URL.revokeObjectURL(imageUrl);
    }
  };
  image.onerror = () => {
    predictionLabel.textContent = "이미지 로드 실패";
    URL.revokeObjectURL(imageUrl);
  };
  image.src = imageUrl;
}

function handleImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    predictionLabel.textContent = "이미지 파일 아님";
    return;
  }

  classifyImage(file);
}

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  handleImageFile(file);
  imageInput.value = "";
});

uploadButton.addEventListener("click", (event) => {
  event.stopPropagation();
  imageInput.click();
});

dropZone.addEventListener("click", () => {
  dropZone.focus();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");

  const [file] = event.dataTransfer.files;
  handleImageFile(file);
});

document.addEventListener("paste", (event) => {
  const items = Array.from(event.clipboardData?.items || []);
  const imageItem = items.find((item) => item.type.startsWith("image/"));
  const imageFile = Array.from(event.clipboardData?.files || []).find((file) => file.type.startsWith("image/"));

  if (!imageItem && !imageFile) {
    return;
  }

  const file = imageFile || imageItem.getAsFile();
  if (file) {
    event.preventDefault();
    handleImageFile(file);
  }
});

loadModel();
