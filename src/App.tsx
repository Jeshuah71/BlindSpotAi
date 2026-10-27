import { useEffect,useRef,useState } from "react";
import type { ChangeEvent } from "react";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import "./App.css";

type Prediction={
  className: string;
  probability: number;
};

type RobustnessPoint = {
  blur: number;
  confidence: number;
  topClass: string;
};


function App() {
  const [image,setImage]=useState<File|null>(null);
  const [imageUrl,setImageUrl]=useState<string|null>(null);
  const [error,setError]=useState("");

  const [model,setModel]=useState<mobilenet.MobileNet|null>(null);
  const [modelLoading,setModelLoading]=useState(true);
  const [modelError,setModelError]=useState("");
  const [predictions,setPredictions]=useState<Prediction[]>([]);
  const [isClassifying,setIsClassifying]=useState(false);
  const [imageReady,setImageReady]=useState(false);
  const imageRef=useRef<HTMLImageElement|null>(null);

  const [blur,setBlur]=useState(0);
  const [brightness,setBrightness]=useState(100);

  const [modifiedPredictions,setModifiedPredictions]=useState<Prediction[]>([]);

  const canvasRef=useRef<HTMLCanvasElement|null>(null);
  const [robustnessResults, setRobustnessResults] = useState<RobustnessPoint[]>([]);

  const [referenceClass, setReferenceClass] = useState("");

  const [isTesting, setIsTesting] = useState(false);

  const [currentBlur, setCurrentBlur] = useState<number | null>(null);

  useEffect(() => {
    let active=true;

    async function loadModel() {
      try {

        // Iniwialize TensorFlow's backend
        await tf.ready();

        console.log("TensorFlow backend:",tf.getBackend());
        // Load the pretrained CNN
        const loadedModel=await mobilenet.load();

        if(active) {
          setModel(loadedModel);
        }
      } catch(error) {
        console.error("Error loading CNN:",error);

        if(active) {
          setModelError("Failed to load the AI model.");
        }
      } finally {
        if(active) {
          setModelLoading(false);
        }
      }


    }

    loadModel();

    return () => {
      active=false;
    };
  },[]);

  useEffect(() => {
    const image=imageRef.current;
    const canvas=canvasRef.current;

    if(!imageReady||!image||!canvas) {
      return;
    }

    if(image.naturalWidth===0||image.naturalHeight===0) {
      return;
    }

    const ctx=canvas.getContext("2d");

    if(!ctx) return;

    // Scale large images down while preserving proportions
    const scale=Math.min(
      1,
      640/image.naturalWidth,
      640/image.naturalHeight
    );

    canvas.width=Math.max(
      1,
      Math.round(image.naturalWidth*scale)
    );

    canvas.height=Math.max(
      1,
      Math.round(image.naturalHeight*scale)
    );

    // Apply the transformations
    ctx.filter=`blur(${blur}px) brightness(${brightness}%)`;

    // Draw the transformed image
    ctx.drawImage(
      image,
      0,
      0,
      canvas.width,
      canvas.height
    );
  },[imageReady,imageUrl,blur,brightness]);

  function handleImageChange(event: ChangeEvent<HTMLInputElement>) {
    const file=event.target.files?.[0];

    if(!file) return;

    // Accept supported image formats
    const supportedTypes=[
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    if(!supportedTypes.includes(file.type)) {
      setError("Please choose a JPG, PNG, or WebP image.");
      return;
    }

    setError("");

    setPredictions([]);
    setModifiedPredictions([]);

    setBlur(0);
    setBrightness(100);

    setImageReady(false);

    setRobustnessResults([]);
    setReferenceClass("");

    // Remember the file
    setImage(file);

    // Create a local preview URL
    setImageUrl(URL.createObjectURL(file));
  }

  // Release old preview URLs when replaced or unmounted
  useEffect(() => {
    return () => {
      if(imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  },[imageUrl]);

  async function classifyImage() {
    // Make sure model and image are available
    if(!model||!imageRef.current||!canvasRef.current||!imageReady||isClassifying) {
      return;
    }

    setIsClassifying(true);
    setError("");

    try {
      // Classify the original immage
      const originalResults=await model.classify(
        imageRef.current,
        3
      );

      // Classify the transformed canvas
      const modifiedResults=await model.classify(
        canvasRef.current,
        3
      );

      // Store both resutls
      setPredictions(originalResults);
      setModifiedPredictions(modifiedResults);

      console.log("Original:",originalResults);
      console.log("Modified:",modifiedResults);

    } catch(error) {
      console.error("Classification error",error);

      setError("Unable to compare these image.");
    } finally {
      setIsClassifying(false);
    }
  }

  async function runRobustnessTest() {
    if (
      !model || !imageRef.current || !imageReady || isClassifying
    ) {
      return;
    }

    setIsTesting(true);
    setIsClassifying(true);
    setError("");
    setRobustnessResults([]);
    setReferenceClass("");

    try {
      const image = imageRef.current;

      // Create a temporary canvas
      const canvas = document.createElement("canvas");

      const scale = Math.min(
        1,
        640 / image.naturalWidth,
        640 / image.naturalHeight
      );

      canvas.width = Math.max(
        1,
        Math.round(image.naturalWidth * scale)
      );

      canvas.height = Math.max(
        1,
        Math.round(image.naturalHeight * scale)
      );

      const ctx = canvas.getContext("2d");

      if (!ctx) {
        throw new Error("Canvas is unavailable");
      }

      const blurLevels = [0, 4, 8, 12, 16, 20];
      const results: RobustnessPoint[] = [];

      let baselineClass = "";

      for (const level of blurLevels) {
        setCurrentBlur(level)
        // Redraw the original image each time
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.filter = `blur(${level}px) brightness(100%)`;

        ctx.drawImage(
          image,
          0,
          0,
          canvas.width,
          canvas.height
        );

        // Analyze the modified image
        const predictions = await model.classify(
          canvas,
          1000
        );

        if (predictions.length === 0) {
          throw new Error("No predictions returned");
        }

        // The first test establishes our reference class
        if (level === 0) {
          baselineClass = predictions[0].className;
        }

        // Find the SAME class at every blur level
        const match = predictions.find(
          (p) => p.className === baselineClass
        );

        results.push({
          blur: level,
          confidence: (match?.probability ?? 0) * 100,
          topClass: predictions[0].className,
        });

        setReferenceClass(baselineClass);

        // Display each completed result immediately
        setRobustnessResults([...results]);

        // Short delay so users can follow the experiment
        await new Promise<void>((resolve) => setTimeout(resolve, 500));

        setCurrentBlur(null);
    }

    setReferenceClass(baselineClass);
    setRobustnessResults(results);

  } catch (error) {
    console.error("Robustness test error:", error);
    setError("Unable to complete robustness test.");
  } finally {
    setIsTesting(false);
    setIsClassifying(false);
  }
}

const firstChangedResult = robustnessResults.find(
  (result) => result.topClass !== referenceClass
);

  return (
    <main className="app">
      <header>
        <p className="eyebrow">COMPUTER VISION PROJECT</p>
        <h1>BlindSpot <span>AI</span></h1>
        <p>
          Discover how image transformations affect
          neural network predictions.
        </p>
      </header>

      <section className="panel">
        <h2>Upload an Image</h2>
        <p>Select an image to begin your experiment.</p>

        <label className="upload-box" htmlFor="image-upload">
          <span>Choose an image</span>
          <small>JPG, PNG, or WebP</small>

          <input
            id="image-upload"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleImageChange}
            disabled={isClassifying}
          />
        </label>

        {error&&<p className="error">{error}</p>}
        {modelError&&<p className="error">Model loading error: {modelError}</p>}

        <p>
          {modelLoading
            ? "Loading neural network..."
            :model
              ? "CNN ready"
              :"CNN unavailable"}
        </p>

        {image && imageUrl && (
          <div className="preview">

            {/* Images side by side */}
            <div className="image-grid">

              {/* Original image */}
              <div className="image-card">
                <h3>Original Image</h3>

                <div className="image-frame">
                  <img
                    ref={imageRef}
                    src={imageUrl}
                    alt="Original"
                    onLoad={() => setImageReady(true)}
                  />
                </div>

                <div className="predictions">
                  <h4>Original Predictions</h4>

                  {predictions.length === 0 && (
                    <p className="prediction-placeholder">
                      Click "Compare Images" to analyze the original image.
                    </p>
                  )}

                  {predictions.map((prediction) => (
                    <div
                      className="prediction"
                      key={prediction.className}
                    >
                      <div className="prediction-header">
                        <span>{prediction.className}</span>
                        <strong>
                          {(prediction.probability * 100).toFixed(1)}%
                        </strong>
                      </div>

                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${prediction.probability * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Modified image */}
              <div className="image-card">
                <h3>Modified Image</h3>

                <div className="image-frame">
                  <canvas ref={canvasRef} />
                </div>

                <div className="predictions">
                  <h4>Modified Predictions</h4>

                  {modifiedPredictions.length === 0 && (
                    <p className="prediction-placeholder">Click "Compare Images" to analyze the current transformation.</p>
                  )}

                  {modifiedPredictions.map((prediction) => (
                    <div
                      className="prediction"
                      key={prediction.className}
                    >
                      <div className="prediction-header">
                        <span>{prediction.className}</span>
                        <strong>
                          {(prediction.probability * 100).toFixed(1)}%
                        </strong>
                      </div>

                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${prediction.probability * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Transformation sliders */}
            <div className="transform-controls">
              <h3>Image Transformations</h3>

              <div className="controls-grid">
                <div className="control-item">
                  <label htmlFor="blur">
                    Blur: {blur}px
                  </label>

                  <input
                    id="blur"
                    type="range"
                    min="0"
                    max="20"
                    value={blur}
                    disabled={isClassifying}
                    onChange={(e) => {
                      setBlur(Number(e.target.value));
                      setModifiedPredictions([]);
                    }}
                  />
                </div>

                <div className="control-item">
                  <label htmlFor="brightness">
                    Brightness: {brightness}%
                  </label>

                  <input
                    id="brightness"
                    type="range"
                    min="20"
                    max="180"
                    value={brightness}
                    disabled={isClassifying}
                    onChange={(e) => {
                      setBrightness(Number(e.target.value));
                      setModifiedPredictions([]);
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Compare button */}
            <button
              className="classify-button compare-button"
              onClick={classifyImage}
              disabled={!model || !imageReady || isClassifying}
            >
              {isClassifying ? "Comparing..." : "Compare Images"}
            </button>

            <button
              className="robustness-button"
              onClick={runRobustnessTest}
              disabled={!model || !imageReady || isClassifying}
            >
              {isTesting
                ? `Testing ${currentBlur ?? 0}px blur...`
                : "Run Automated Robustness Test"
              }

            </button>

            <p className="experiment-helper">
              <strong>Compare Images:</strong> Classifies the original
              and the modified image using your selected settings.
              <br />
              <strong>Automated Robustness Test:</strong> Tests six
              blur levels while keeping brightness at 100%.
            </p>

            {robustnessResults.length > 0 && (
              <div className="robustness-panel">
                <h3>AI Robustness Analysis</h3>

                <p>
                  Tracking confidence for:
                  <strong> {referenceClass}</strong>
                </p>

                {robustnessResults.map((result) => (
                  <div className="robustness-row" key={result.blur}>

                    <span>{result.blur}px</span>

                    <div className="robustness-track">
                      <div
                        className="robustness-fill"
                        style={{
                          width: `${result.confidence}%`,
                        }}
                      />
                    </div>

                    <strong>
                      {result.confidence.toFixed(1)}%
                    </strong>

                    {/* Show when the CNN changes the prediction */}
                    {firstChangedResult?.blur === result.blur && (
                      <small className="prediction-changed">
                        First prediction change at {result.blur}px:
                        {" "}{result.topClass}
                      </small>
                    )}
                  </div>
                ))}
                {!isTesting && robustnessResults.length > 0 && (
                  <div className="robustness-summary">
                    <h4>Experiment Conclusion</h4>

                    {firstChangedResult ? (
                      <p>
                        The CNN first changed its top prediction at{" "}
                        {firstChangedResult.blur}px blur.

                        New prediction:{" "}
                        <strong>{firstChangedResult.topClass}</strong>
                      </p>
                    ) : (
                      <p>
                        The CNN maintained the same top predicted class
                        across all tested blur levels.
                      </p>
                    )}
                  </div>
                )}
                <p className="robustness-note">
                  These scores track the same classification
                  across different blur levels.
                </p>
              </div>
            )}


            {/* Experiment results */}
            {predictions.length > 0 &&
              modifiedPredictions.length > 0 && (
                <div className="comparison-result">
                  <h3>Current Image Comparison</h3>

                  <p>
                    <strong>Original:</strong>{" "}
                    {predictions[0].className} (
                    {(predictions[0].probability * 100).toFixed(1)}%)
                  </p>

                  <p>
                    <strong>Modified:</strong>{" "}
                    {modifiedPredictions[0].className} (
                    {(modifiedPredictions[0].probability * 100).toFixed(1)}%)
                  </p>

                  <p>
                    {predictions[0].className ===
                    modifiedPredictions[0].className
                      ? "The top prediction remained the same."
                      : "The top prediction changed."}
                  </p>
                </div>
              )}
          </div>
        )}
      </section>
    </main>
  );
}

export default App;

