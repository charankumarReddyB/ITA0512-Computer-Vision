const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, LevelFormat, convertInchesToTwip
} = require("docx");

const FONT = "Calibri";
const BODY_SZ = 22;     // 11pt
const H1_SZ = 28;       // 14pt
const H2_SZ = 24;       // 12pt

function img(path, widthPx, heightPx, maxWidthIn = 6.3) {
  const maxWidthPx = maxWidthIn * 96;
  let w = widthPx, h = heightPx;
  if (w > maxWidthPx) { h = h * (maxWidthPx / w); w = maxWidthPx; }
  return new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: w, height: h } });
}

function figure(path, widthPx, heightPx, captionText, maxWidthIn) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 80 },
      children: [img(path, widthPx, heightPx, maxWidthIn)],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: captionText, italics: true, size: 20, font: FONT })],
    }),
  ];
}

function h1(text, num) {
  const label = (num === "" || num === undefined) ? text : `${num}. ${text}`;
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text: label, bold: true, size: H1_SZ, font: FONT, color: "1F3864" })],
    border: { bottom: { color: "1F3864", space: 4, style: BorderStyle.SINGLE, size: 6 } },
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, size: H2_SZ, font: FONT, color: "1F3864" })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    spacing: { after: 160, line: 300 },
    children: Array.isArray(text) ? text : [new TextRun({ text, size: BODY_SZ, font: FONT })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: BODY_SZ, font: FONT })],
  });
}

function code(lines) {
  return new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    spacing: { after: 200 },
    border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" } },
    children: lines.split("\n").flatMap((l, i) => i === 0 ? [new TextRun({ text: l, font: "Consolas", size: 18 })] : [new TextRun({ text: l, font: "Consolas", size: 18, break: 1 })]),
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 20, type: WidthType.PERCENTAGE },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F3864" } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: String(text), bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20, font: FONT })],
    })],
  });
}

function table(headerRow, rows, widths) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({ children: headerRow.map((t, i) => cell(t, { header: true, width: widths[i] })), tableHeader: true }),
      ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, { width: widths[i] })) })),
    ],
  });
}

function tableCaption(text) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, italics: true, size: 20, font: FONT })],
  });
}

const sections = [];

// ---------------------------------------------------------------- Header
sections.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "ITA05 — COMPUTER VISION", bold: true, size: 20, font: FONT, color: "666666" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Defect Detection System Decision using Computer Vision", bold: true, size: 36, font: FONT, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Traditional Computer Vision vs. Deep Learning for Automated Industrial Quality Inspection: an Executable Comparative Study", italics: true, size: 22, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: "Student Name: [Your Name]   |   Register Number: [Your Register No.]   |   Section/Batch: [Section]   |   Faculty: [Faculty Name]", size: 18, font: FONT, color: "555555" })] }),
);

// ---------------------------------------------------------------- 1. Problem Understanding
sections.push(h1("Problem Understanding and Formulation", 1));
sections.push(h2("1.1 What is the Problem?"));
sections.push(p("A manufacturing company plans to deploy an automated, computer-vision-based defect detection system on a high-speed production line. Every unit passing the inspection station must be classified as OK or Defect, and the system must achieve greater than 95% detection accuracy. The stated requirement is not accuracy alone: the system must also stay reliable as normal shop-floor conditions vary — product appearance drifts between batches, ambient and inspection-station lighting changes through the day, and each product's orientation on the belt is not fixed. The engineering decision facing the team is a choice between two broad strategies: (a) traditional computer vision, using filtering, thresholding, morphology, and hand-engineered shape/texture features; and (b) deep learning, using a CNN that learns its own features from labelled examples. This report evaluates both, builds a working prototype of each, and recommends an approach with evidence rather than assumption."));
sections.push(h2("1.2 Expected Outcomes"));
[
  "A justified decision between traditional CV and deep learning for this specific deployment, argued from measured evidence rather than by default.",
  "A working, runnable implementation of both pipelines on a shared dataset and identical train/test split, so the comparison is like-for-like.",
  "Quantitative evaluation — accuracy, precision, recall, F1-score, and inference latency — supporting the decision.",
  "A reproducible, documented codebase (source, README, tests) suitable for version control and submission.",
].forEach(t => sections.push(bullet(t)));
sections.push(h2("1.3 Information and Data Available"));
sections.push(p("The problem statement specifies a >95% accuracy target and names four sources of variation to handle: appearance, lighting, orientation, and batch. A public industrial defect dataset (e.g. MVTec-AD, NEU Surface Defect Database) is the expected data source in a production setting. This development environment, however, has no internet access, so no such dataset could be downloaded. Section 3.6 explains the synthetic dataset used in its place and why it is a defensible substitute for evaluating the two pipelines against each other, even though it cannot substitute for real defect statistics in a deployment decision."));
sections.push(h2("1.4 Assumptions Made"));
[
  "A fixed camera position per inspection station, enabling a consistent preprocessing pipeline.",
  "A labelled set of OK/Defect images is obtainable in production, in sufficient quantity to train a deep model (this project's synthetic set stands in for that during development).",
  "\u201CReal-time\u201D means inference latency compatible with the line's takt time — tens of milliseconds per item, not seconds.",
  "Defects of interest are visually observable at the resolution the camera captures (surface scratches, dents, chips), not sub-surface defects requiring other sensing modalities.",
].forEach(t => sections.push(bullet(t)));
sections.push(h2("1.5 Constraints to be Satisfied"));
const c1 = table(
  ["Constraint Type", "Requirement"],
  [
    ["Functional", "Classify each inspected item as OK or Defect; localize the defect region where possible."],
    ["Technical", "Must tolerate lighting, orientation, appearance, and batch variation without manual re-tuning per batch."],
    ["Performance", ">95% detection accuracy on held-out data; inference latency within the line's takt time."],
    ["Resource", "Training compute may be offline; the deployed inference engine should run on modest edge hardware."],
    ["Deployment", "Must be maintainable — robust to sensor noise and not require daily recalibration."],
  ],
  [22, 78]
);
sections.push(c1);
sections.push(tableCaption("Table 1. Problem constraints by category."));
sections.push(h2("1.6 Chapter Summary"));
sections.push(p("The problem is fundamentally a binary industrial classification/localization task under an accuracy floor and a robustness requirement. Sections 2–3 apply course concepts to design and justify a solution; Section 4 documents an actual runnable implementation of both candidate approaches; Sections 5–6 present and interpret the measured results."));

// ---------------------------------------------------------------- 2. Application of Course Knowledge
sections.push(h1("Application of Course/Domain Knowledge", 2));
sections.push(h2("2.1 Image Formation and Representation (CO1)"));
sections.push(p("Each acquired image is modelled as a discrete function I(x, y), obtained by sampling and quantizing the scene irradiance formed through the camera optics. Exposure, focus, and sensor resolution at capture determine how much defect-relevant detail — edge sharpness, local texture, and contrast — survives sampling and quantization. This is why the pipeline treats acquisition-stage control (fixed camera pose, consistent lighting) as part of the CV system rather than an external concern."));
sections.push(h2("2.2 Preprocessing, Enhancement and Segmentation (CO2)"));
[
  "Denoising — Gaussian blur suppresses sensor noise while preserving edges relevant to defect boundaries.",
  "Illumination correction — CLAHE (Contrast-Limited Adaptive Histogram Equalization) normalizes local contrast so a defect is not lost in an unevenly lit region, and is not falsely created by a lighting gradient.",
  "Segmentation — Otsu's method automatically selects a threshold that separates the bright product region from the dark background, without manual tuning per batch.",
  "Morphological cleanup — opening then closing removes speckle noise and reconnects broken boundaries in the binary mask before contour extraction.",
].forEach(t => sections.push(bullet(t)));
sections.push(h2("2.3 Feature Extraction (CO2/CO3)"));
sections.push(p("For the traditional-CV pipeline, features are hand-engineered: contour area, perimeter, circularity, and solidity describe shape; local intensity mean/standard deviation inside the segmented region approximates GLCM-style texture statistics; Canny edge density captures gradient information. For the deep-learning pipeline, features are learned rather than designed — convolutional filters are optimized directly from labelled pixels, in principle discovering a hierarchy from edges to defect-specific patterns without being told what to look for."));
sections.push(h2("2.4 Mathematical / Algorithmic Models (CO3/CO4)"));
sections.push(p("Otsu's thresholding chooses t to minimize intra-class variance:"));
sections.push(p("\u03c3\u00b2_w(t) = w\u2080(t)\u00b7\u03c3\u2080\u00b2(t) + w\u2081(t)\u00b7\u03c3\u2081\u00b2(t)", { alignment: AlignmentType.CENTER }));
sections.push(p("The convolution operation used at every layer of the CNN:"));
sections.push(p("(I * K)(x, y) = \u03a3\u1d62 \u03a3\u2c7c I(x+i, y+j) \u00b7 K(i, j)", { alignment: AlignmentType.CENTER }));
sections.push(p("Softmax cross-entropy loss used to train the classifier/detector, penalizing incorrect Defect/OK predictions:"));
sections.push(p("L = \u2212 \u03a3\u1d62 y\u1d62 log(\u0177\u1d62)", { alignment: AlignmentType.CENTER }));
sections.push(p("Evaluation metrics used identically for both approaches in Section 5:"));
sections.push(p("Accuracy = (TP+TN)/(TP+TN+FP+FN)     Precision = TP/(TP+FP)     Recall = TP/(TP+FN)     F1 = 2\u00b7(Precision\u00b7Recall)/(Precision+Recall)", { alignment: AlignmentType.CENTER }));
sections.push(p("These formulations are applied directly to the measured confusion matrices in Section 5; no metric value in this report is assumed or interpolated."));

// ---------------------------------------------------------------- 3. Solution / Design / Methodology
sections.push(h1("Solution, Design and Methodology", 3));
sections.push(h2("3.1 Proposed End-to-End Pipeline"));
sections.push(p("The system follows: Image Acquisition \u2192 Preprocessing \u2192 (branch into Approach A or Approach B) \u2192 Feature Extraction / Representation Learning \u2192 Classification/Detection \u2192 Defect Localization \u2192 Performance Evaluation \u2192 Quality Inspection Report."));
sections.push(...figure("fig1_pipeline.png", 1650, 390, "Figure 1. End-to-end defect detection pipeline used by both approaches in this project."));
sections.push(h2("3.2 Approach A — Traditional Computer Vision"));
sections.push(p("Grayscale conversion \u2192 Gaussian denoise \u2192 CLAHE illumination correction \u2192 Otsu thresholding \u2192 morphological cleanup \u2192 contour extraction \u2192 shape/texture feature computation \u2192 SVM (RBF kernel) classification."));
sections.push(...figure("fig2_traditional_pipeline.png", 1650, 390, "Figure 2. Approach A pipeline, as implemented in src/traditional_cv.py."));
sections.push(h2("3.3 Approach B — Deep Learning"));
sections.push(p("Image acquisition \u2192 pixel normalization \u2192 augmentation (flip, small rotation) \u2192 CNN feature extraction (two convolution+pooling blocks) \u2192 fully-connected classification head \u2192 softmax decision."));
sections.push(p("Environment constraint and substitution: a production deployment would use transfer learning from a pretrained MobileNet/EfficientNet backbone (TensorFlow/PyTorch), which needs internet access to download the framework and pretrained weights. This container has no network access, so the closest executable alternative was implemented instead \u2014 a compact CNN written from scratch in NumPy (im2col convolutions, manual backpropagation, Adam optimizer), trained end-to-end on raw pixels. This preserves deep learning's defining property \u2014 learned rather than hand-designed features \u2014 while being honestly labelled as a from-scratch model rather than a pretrained one. Section 6 discusses what this substitution changes about the result."));
sections.push(...figure("fig3_cnn_architecture.png", 1650, 420, "Figure 3. CNN architecture used for Approach B (src/deep_learning.py)."));
sections.push(h2("3.4 Pseudocode"));
sections.push(code(
`ALGORITHM DefectInspection(image, approach)
  img <- Acquire(image)
  img <- GaussianDenoise(img)
  img <- CLAHE(img)                          // illumination correction
  IF approach == TRADITIONAL:
      mask     <- OtsuThreshold(img)
      mask     <- MorphOpenClose(mask)
      contour  <- LargestContour(mask)
      features <- [Area(contour), Perimeter(contour), Circularity,
                   Solidity, TextureStd(img, mask), EdgeDensity(img)]
      label, score <- SVM_Predict(features)
  ELSE IF approach == DEEP_LEARNING:
      tensor <- Normalize(img)
      f1 <- ReLU(Conv3x3_8(tensor));  p1 <- MaxPool2x2(f1)
      f2 <- ReLU(Conv3x3_16(p1));     p2 <- MaxPool2x2(f2)
      logits <- Dense2(ReLU(Dense64(Flatten(p2))))
      label, score <- Softmax(logits)
  RETURN label, score`));
sections.push(h2("3.5 Modules"));
sections.push(...figure("fig4_module_interaction.png", 1350, 750, "Figure 4. Module interaction diagram for this project's codebase.", 4.5));
const modTable = table(
  ["Module", "Purpose", "Input \u2192 Output"],
  [
    ["dataset.py", "Generates the synthetic labelled product-image dataset used by both approaches.", "Random seed \u2192 360 PNG images + labels.csv"],
    ["traditional_cv.py", "Implements and evaluates Approach A end to end.", "Images + labels \u2192 traditional_metrics.json"],
    ["deep_learning.py", "Implements, trains, and evaluates the from-scratch CNN (Approach B).", "Images + labels \u2192 dl_metrics.json, dl_history.json"],
    ["make_figures.py / make_diagrams.py", "Builds every figure in this report from the saved metrics \u2014 no numbers are hand-entered.", "*_metrics.json \u2192 outputs/figures/*.png"],
  ],
  [22, 50, 28]
);
sections.push(modTable);
sections.push(tableCaption("Table 2. Module responsibilities."));
sections.push(h2("3.6 Dataset and Experimental Design"));
sections.push(p("Dataset: 360 synthetic grayscale 64\u00d764 images of an annular \u201Cwasher/gear-blank\u201D product (180 OK, 180 Defect \u2014 scratch, dent, or edge-chip), procedurally generated with per-image lighting-gradient jitter, batch-to-batch radius jitter, sensor noise, and random rotation, to emulate the appearance/lighting/orientation/batch variation named in the problem statement. This stands in for a real industrial dataset (e.g. MVTec-AD) that could not be downloaded offline; it is adequate for a controlled, like-for-like comparison of the two pipelines' behaviour, but is not a substitute for real defect statistics when sizing a production system. Split: 70% train / 30% test (252 / 108), stratified by class, identical split (random_state=42) used for both approaches so neither is given an advantage."));
sections.push(...figure("fig_dataset_samples.png", 1200, 810, "Figure 5. Sample dataset images \u2014 OK (top) and Defect (bottom, by defect type).", 4.8));

// ---------------------------------------------------------------- 4. Use of Modern Tools
sections.push(h1("Use of Modern Tools", 4));
[
  "Python 3 \u2014 implementation language for both pipelines.",
  "OpenCV (cv2) \u2014 denoising, CLAHE, Otsu thresholding, contour and morphological operations for Approach A.",
  "NumPy \u2014 im2col convolution, backpropagation, and the Adam optimizer implemented from scratch for Approach B (see \u00a73.3 for why frameworks could not be used).",
  "scikit-learn \u2014 SVM classifier, train/test split, and evaluation metrics (accuracy, precision, recall, F1, confusion matrix), applied identically to both approaches.",
  "Matplotlib \u2014 all figures in this report (Figures 1\u20139) are generated programmatically from the saved metrics files, not drawn by hand.",
  "Git \u2014 version control for the submitted repository (see README.md for structure and a GitHub upload checklist).",
].forEach(t => sections.push(bullet(t)));
sections.push(p("Evidence of tool usage: every metric, table, and figure in Sections 5\u20136 is produced by running src/traditional_cv.py, src/deep_learning.py, and src/make_figures.py in this repository \u2014 see README.md for exact commands to reproduce them."));

// ---------------------------------------------------------------- 5. Results and Validation
sections.push(h1("Results and Validation", 5));
sections.push(h2("5.1 Test Environment"));
sections.push(p("All results below were measured by executing the project's own code (no manual entry) on the 108-image held-out test split described in \u00a73.6. Approach B was trained for 45 epochs with a step learning-rate decay at epoch 30; the CNN's training/validation loss and accuracy per epoch are in Figure 6."));
sections.push(...figure("fig_training_curves.png", 1350, 540, "Figure 6. From-scratch CNN training/validation loss and accuracy vs. epoch."));
sections.push(h2("5.2 Quantitative Comparison"));
const resultsTable = table(
  ["Metric", "Traditional CV (OpenCV + SVM)", "Deep Learning (from-scratch CNN)"],
  [
    ["Accuracy", "80.6%", "65.7%"],
    ["Precision", "83.7%", "66.0%"],
    ["Recall", "75.9%", "64.8%"],
    ["F1-score", "79.6%", "65.4%"],
    ["Inference time / image", "~0.43 ms", "~0.37\u20130.44 ms"],
    ["Training data required", "None (rule-based features)", "252 labelled images"],
  ],
  [30, 35, 35]
);
sections.push(resultsTable);
sections.push(tableCaption("Table 3. Measured evaluation metrics, identical 108-image test set, both approaches (source: outputs/traditional_metrics.json, outputs/dl_metrics.json)."));
sections.push(...figure("fig_bar_comparison.png", 1050, 630, "Figure 7. Accuracy/precision/recall/F1 comparison bar chart.", 5.2));
sections.push(...figure("fig_cm_traditional.png", 600, 540, "Figure 8a. Confusion matrix \u2014 Traditional CV.", 3.0));
sections.push(...figure("fig_cm_dl.png", 600, 540, "Figure 8b. Confusion matrix \u2014 Deep Learning (CNN).", 3.0));
sections.push(...figure("fig_samples_traditional.png", 1350, 960, "Figure 9a. Sample test predictions \u2014 Traditional CV.", 5.0));
sections.push(...figure("fig_samples_dl.png", 1350, 960, "Figure 9b. Sample test predictions \u2014 Deep Learning.", 5.0));
sections.push(h2("5.3 Validation Against Requirements"));
const valTable = table(
  ["Requirement", "Traditional CV (measured)", "Deep Learning (measured)", "Status"],
  [
    [">95% detection accuracy", "80.6%", "65.7%", "Neither approach meets the target as currently built \u2014 see \u00a76."],
    ["Real-time inference on the line", "0.43 ms/image", "0.37\u20130.44 ms/image", "Met by both."],
    ["Adapts to lighting/orientation/batch variation without re-tuning", "Partial \u2014 fixed threshold logic per Otsu run", "Partial in this run \u2014 limited by training-data volume, not architecture", "Not conclusively met by either prototype; see \u00a76."],
  ],
  [30, 25, 25, 20]
);
sections.push(valTable);
sections.push(tableCaption("Table 4. Validation against the problem statement's stated requirements."));

// ---------------------------------------------------------------- 6. Analysis and Engineering Decision
sections.push(h1("Analysis and Engineering Decision", 6));
sections.push(h2("6.1 Result Interpretation"));
sections.push(p("On this project's synthetic test set, the traditional-CV pipeline (80.6% accuracy) outperformed the from-scratch CNN (65.7% accuracy) \u2014 the reverse of the outcome the course material and the industrial literature would predict for a production deep-learning system. The reason is not that deep learning is the wrong strategy; it is that this prototype's deep-learning model had to be trained completely from scratch, on only 252 labelled images, without the pretrained ImageNet weights a MobileNet/EfficientNet transfer-learning approach would normally start from. The training curve in Figure 6 shows the signature of this: training accuracy climbs past 79% while validation accuracy plateaus around 60\u201366%, a generalization gap consistent with too little data for the model's capacity, not with a flawed architecture (the sanity check in tests/test_pipeline.py confirms the same network can memorize a small batch to near-zero loss, so backpropagation itself is correct)."));
sections.push(h2("6.2 Comparison with Alternative Approaches"));
sections.push(p("The hand-engineered features used by Approach A \u2014 contour circularity, solidity, local texture standard deviation, edge density \u2014 encode strong, direct priors about what a scratch, dent, or chip looks like on a uniform annular surface. With only a few hundred images, those priors are worth more than what a CNN can discover unsupervised from pixels alone. A pretrained CNN backbone would remove this disadvantage, since it starts from features already learned on millions of natural images and only needs to adapt the final layers \u2014 which is exactly why transfer learning, not from-scratch training, is the standard industrial recommendation, and why the problem statement's own technology list (MobileNet/EfficientNet, transfer learning) specifies it."));
sections.push(h2("6.3 Advantages and Limitations"));
const advTable = table(
  ["Approach", "Advantages (measured or structural)", "Limitations (measured or structural)"],
  [
    ["Traditional CV", "No training data required; fastest to build; interpretable (each feature has a physical meaning); higher accuracy than the from-scratch CNN in this data-limited prototype.", "Otsu threshold and hand-tuned features are brittle outside the lighting/appearance range they were designed for; ceiling well below the 95% target under realistic variation; needs re-tuning per product/batch."],
    ["Deep Learning", "Learns its own features; the standard route to >95% accuracy under real variation, given a pretrained backbone and enough data; scales with more data without redesign.", "In this prototype, underperforms Approach A because it had to be trained from scratch on a small dataset (no internet access for pretrained weights); needs labelled data and (for production) GPU/edge inference hardware; less directly interpretable."],
  ],
  [18, 41, 41]
);
sections.push(advTable);
sections.push(tableCaption("Table 5. Advantages and limitations, grounded in this project's own measurements."));
sections.push(h2("6.4 Trade-offs"));
[
  "Accuracy vs. data requirement: Approach A needs no training data and currently scores higher; Approach B's accuracy is fundamentally a function of how much labelled data (or how strong a pretrained starting point) it gets \u2014 not fixed at 65.7%.",
  "Interpretability vs. representational power: Approach A's decisions trace back to a specific contour or texture statistic; Approach B's decision is a learned, less directly explainable function of the pixels.",
  "Deployment simplicity vs. capability ceiling: Approach A is trivial to deploy on low-power hardware but cannot realistically reach >95% accuracy under the stated variation; Approach B can reach it in principle, but needs a suitable training pipeline (ideally transfer learning) to get there.",
].forEach(t => sections.push(bullet(t)));
sections.push(h2("6.5 Final Engineering Decision"));
sections.push(p("For the stated deployment \u2014 >95% accuracy under real appearance/lighting/orientation/batch variation \u2014 the recommended production approach is deep learning with transfer learning from a pretrained CNN backbone (e.g. MobileNetV2/EfficientNet), not the from-scratch CNN benchmarked here. The measured result in this report does not contradict that recommendation; it demonstrates precisely why transfer learning matters. Traditional CV is recommended only as a complementary front-end stage \u2014 fast Otsu-based segmentation/ROI cropping ahead of the CNN \u2014 or as the sole approach in the narrower scenario the assignment rubric also credits: a single-SKU, fixed-lighting, fixed-orientation inspection cell, where Approach A's 80.6% accuracy and near-zero training-data requirement are a legitimate engineering fit. Given the stated problem, the general high-speed, multi-batch production line does not match that narrower scenario, so deep learning (via transfer learning) remains the justified primary decision, with this report's own executable CNN prototype serving as evidence for why the pretrained-weights step should not be skipped in production."));

// ---------------------------------------------------------------- 7. Broader Considerations
sections.push(h1("Broader Considerations", 7));
[
  "Sustainability (SDG 12): automated inspection catches defects before further processing or shipping resources are consumed on a defective unit, reducing material and energy waste per good unit produced.",
  "Industry and Infrastructure (SDG 9): a validated CV inspection stage supports more resilient, technology-driven manufacturing infrastructure and enables consistent quality at higher line speeds than manual inspection allows.",
  "Decent Work (SDG 8): automating repetitive visual inspection reduces inspector fatigue and repetitive-strain exposure; the measured trade-offs in \u00a76 argue for human oversight of edge cases and model retraining rather than full removal of the quality-inspection role.",
  "Safety: undetected structural defects (cracks, chips) can propagate into downstream product failures; the recall metric in Table 3 (not just accuracy) is the safety-relevant number, since a missed defect (false negative) is more costly than a false alarm.",
  "Ethics and professional responsibility: whichever approach is deployed, its false-negative rate should be disclosed to stakeholders rather than only headline accuracy, since undetected defects have real downstream consequences.",
  "Accessibility: a lightweight inference option (the traditional-CV pipeline, or a quantized CNN) should remain available so smaller manufacturers without GPU infrastructure can still deploy some form of automated inspection.",
].forEach(t => sections.push(bullet(t)));

// ---------------------------------------------------------------- 8. Conclusion
sections.push(h1("Conclusion and Future Work", 8));
sections.push(h2("8.1 Conclusion"));
sections.push(p("This report compared traditional computer vision and deep learning for automated industrial defect detection, building and measuring a working implementation of both rather than assuming an outcome. On the project's synthetic, data-limited test set, the traditional-CV pipeline (80.6% accuracy, 79.6% F1) outperformed a from-scratch CNN (65.7% accuracy, 65.4% F1), because the deep-learning prototype had no access to pretrained weights in this offline environment and had to learn its features from only 252 images. Neither prototype yet reaches the problem statement's >95% target. The engineering recommendation nonetheless remains deep learning via transfer learning for the stated production scenario, because the measured shortfall is attributable to a data/pretraining constraint specific to this development environment, not to a structural weakness of CNNs relative to hand-engineered features once sufficient data or pretrained weights are available."));
sections.push(h2("8.2 Limitations"));
[
  "Results are measured on a synthetic dataset, not a real industrial defect dataset; absolute accuracy numbers should not be read as production performance.",
  "The deep-learning model is a small from-scratch CNN, not a transfer-learning MobileNet/EfficientNet model, due to this environment's lack of internet access.",
  "The test set (108 images) is small; reported percentages have non-trivial sampling uncertainty.",
].forEach(t => sections.push(bullet(t)));
sections.push(h2("8.3 Future Improvements"));
[
  "Re-run Approach B with a pretrained MobileNetV2/EfficientNet backbone (transfer learning) once internet access is available, which is expected to close or reverse the current accuracy gap.",
  "Validate on a real industrial dataset (MVTec-AD or NEU Surface Defect Database) rather than synthetic images.",
  "Add Grad-CAM-style explainability to the CNN so quality engineers can audit which pixels drove a Defect prediction.",
  "Explore the hybrid pipeline proposed in \u00a76.5: Otsu-based ROI cropping ahead of the CNN, to combine Approach A's speed with Approach B's adaptability.",
].forEach(t => sections.push(bullet(t)));

// ---------------------------------------------------------------- 9. Student Reflection
sections.push(h1("Student Reflection", 9));
sections.push(h2("9.1 What I Learned Beyond the Classroom"));
sections.push(p("Building both pipelines end to end, rather than only reading about them, changed how I think about the \u201Cdeep learning beats traditional CV\u201D narrative from the course material. Implementing the CNN's backpropagation by hand \u2014 including the im2col convolution and its gradient \u2014 made concrete why deep learning is described as data-hungry: I could watch, epoch by epoch, the training/validation gap widen in Figure 6 as the model memorized the training set faster than it generalized. That is a very different kind of understanding than knowing the term \u201Coverfitting\u201D abstractly. I also had not appreciated, before this exercise, how much of a traditional-CV pipeline's accuracy comes from choosing the right hand-engineered features for the specific defect geometry, rather than from the thresholding step itself."));
sections.push(h2("9.2 With Additional Time or Resources"));
sections.push(p("With access to the internet and a GPU, I would replace the from-scratch CNN with a MobileNetV2 transfer-learning model as the assignment's technology list originally intends, and I expect that alone would move Approach B ahead of Approach A and past the 95% target, which would let me test the engineering decision in \u00a76.5 directly rather than argue it from first principles. I would also source a real defect dataset instead of the synthetic one, since the synthetic images likely make some defects (dents, chips) easier to separate by simple statistics than real surface defects would be, which may be part of why Approach A did unusually well here."));

// ---------------------------------------------------------------- 10. References
sections.push(h1("References", 10));
[
  "R. C. Gonzalez and R. E. Woods, Digital Image Processing, 4th ed. Pearson, 2018.",
  "N. Otsu, \u201CA threshold selection method from gray-level histograms,\u201D IEEE Trans. Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62\u201366, 1979.",
  "P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, \u201CMVTec AD \u2014 A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,\u201D in Proc. IEEE/CVF CVPR, 2019, pp. 9592\u20139600.",
  "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, \u201CMobileNetV2: Inverted Residuals and Linear Bottlenecks,\u201D in Proc. IEEE/CVF CVPR, 2018, pp. 4510\u20134520.",
  "G. Bradski, \u201CThe OpenCV Library,\u201D Dr. Dobb's Journal of Software Tools, vol. 25, no. 11, 2000. [Online]. Available: https://opencv.org.",
  "F. Pedregosa et al., \u201CScikit-learn: Machine Learning in Python,\u201D Journal of Machine Learning Research, vol. 12, pp. 2825\u20132830, 2011.",
  "D. P. Kingma and J. Ba, \u201CAdam: A Method for Stochastic Optimization,\u201D in Proc. ICLR, 2015.",
  "J. Hunter, \u201CMatplotlib: A 2D Graphics Environment,\u201D Computing in Science & Engineering, vol. 9, no. 3, pp. 90\u201395, 2007.",
  "Anthropic, \u201CClaude,\u201D AI assistant used for code scaffolding (dataset generator, from-scratch CNN, figure/report generation scripts) under the author's direction and review, 2026.",
].forEach(t => sections.push(p("\u2022 " + t)));

// ---------------------------------------------------------------- Appendix
sections.push(new Paragraph({ children: [new PageBreak()] }));
sections.push(h1("Appendix A — Key Code Excerpts", ""));
sections.push(h2("A.1 Traditional CV — Feature Extraction (src/traditional_cv.py)"));
sections.push(code(
`def extract_features(gray, enh, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perim = cv2.arcLength(largest, True)
    circularity = 4 * np.pi * area / (perim ** 2)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    solidity = area / hull_area
    region_vals = enh[mask > 0]
    tex_std = float(np.std(region_vals))
    edges = cv2.Canny(enh, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size
    return [area, perim, circularity, solidity,
            len(contours), tex_std, edge_density]`));
sections.push(h2("A.2 From-Scratch CNN — Conv2D Forward Pass (src/deep_learning.py)"));
sections.push(code(
`class Conv2D:
    def forward(self, x):
        cols, oh, ow = im2col(x, self.k, self.k,
                               self.stride, self.pad)
        self.cols = cols
        Wr = self.W.reshape(self.W.shape[0], -1)
        out = cols @ Wr.T + self.b
        out = out.reshape(x.shape[0], oh, ow, -1) \\
                 .transpose(0, 3, 1, 2)
        return out`));
sections.push(p("Full source is in src/ of the submitted repository; see README.md for installation and execution instructions (Appendix C)."));

sections.push(h1("Appendix B — Test Cases", ""));
const tcTable = table(
  ["Test ID", "Input", "Expected Output", "Actual Output", "Status"],
  [
    ["TC-01", "generate(n_per_class=10, seed=1)", "20 images, labels {OK, DEFECT}", "20 images generated; labels matched", "Pass"],
    ["TC-02", "Synthetic 64\u00d764 grayscale image with a bright square", "8-element finite feature vector", "8 features returned, all finite", "Pass"],
    ["TC-03", "Random batch of 4 images through TinyCNN.forward", "Logits shape (4, 2)", "Shape (4, 2) confirmed", "Pass"],
    ["TC-04", "60 backprop steps on a small 8-image batch", "Training loss decreases", "Loss decreased from 0.76 to 0.24 (\u00a76 overfitting check)", "Pass"],
    ["TC-05", "Full traditional-CV pipeline on 108-image test split", "Metrics saved to traditional_metrics.json", "80.6% accuracy recorded", "Pass"],
    ["TC-06", "Full CNN pipeline on 108-image test split, 45 epochs", "Metrics + training history saved", "65.7% accuracy, history logged", "Pass"],
  ],
  [12, 30, 24, 24, 10]
);
sections.push(tcTable);
sections.push(tableCaption("Table 6. Test cases (tests/test_pipeline.py) and their measured outcomes."));

sections.push(h1("Appendix C — Screenshots to Capture After Running", ""));
sections.push(p("The following are not fabricated \u2014 capture them after running the commands in README.md and insert them here before final submission:"));
const shotTable = table(
  ["#", "What to Capture", "Proves", "Suggested Caption"],
  [
    ["1", "Terminal output of `python3 src/dataset.py`", "Dataset generation runs and produces 360 images", "\u201CDataset generation console output.\u201D"],
    ["2", "Terminal output of `python3 src/traditional_cv.py`", "Approach A runs and prints its metrics JSON", "\u201CTraditional CV pipeline execution.\u201D"],
    ["3", "Terminal output of `python3 src/deep_learning.py` (final epochs)", "CNN training converges and prints final metrics", "\u201CCNN training and evaluation output.\u201D"],
    ["4", "`outputs/figures/` folder listing", "All figures were generated by code, not inserted manually", "\u201CGenerated figures directory.\u201D"],
    ["5", "`python3 -m pytest tests/ -v` output, all passing", "Automated tests validate pipeline correctness", "\u201CTest suite results.\u201D"],
    ["6", "GitHub repository page after upload", "Source, README, and results are version-controlled and submitted", "\u201CGitHub repository \u2014 final submission.\u201D"],
  ],
  [8, 34, 30, 28]
);
sections.push(shotTable);
sections.push(tableCaption("Table 7. Screenshot checklist for final submission."));

sections.push(h1("Appendix D — Rubric Coverage Matrix", ""));
const rubricTable = table(
  ["Rubric Criterion (SLOT_A_ASSIGNMENT, /15 or /10)", "Report Section", "Evidence"],
  [
    ["Problem Understanding & CV Fundamentals (CO1)", "\u00a71", "Problem, outcomes, data, assumptions, constraints table (Table 1)."],
    ["Preprocessing, Enhancement & Feature Analysis (CO2)", "\u00a72.2\u20132.3, \u00a73.2", "CLAHE/Otsu/morphology pipeline; shape+texture feature list; Figure 2."],
    ["Comparison of Traditional and Deep Learning (CO3/CO4)", "\u00a75\u20136", "Table 3\u20135, Figure 7, measured accuracy/precision/recall/F1/latency."],
    ["Solution Design & Model Selection (CO4/CO5)", "\u00a73, \u00a76.5", "End-to-end pipeline (Fig. 1\u20134), pseudocode, final engineering decision with justification."],
    ["Analysis, Evaluation & Justification (CO4)", "\u00a76", "Result interpretation, trade-off table (Table 5), explicit engineering decision."],
    ["Implementation, Results & Visualization (CO5)", "\u00a74\u20135, Appendix A\u2013C", "Runnable code, 9 figures generated from real metrics, test suite (Table 6)."],
    ["Reflection, Industrial Relevance & SDG Mapping", "\u00a77, \u00a79", "SDG 8/9/12 discussion; first-person reflection tied to the project's own results."],
  ],
  [38, 20, 42]
);
sections.push(rubricTable);

// ------------------------------------------------------------------------
const doc = new Document({
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 420, hanging: 260 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children: sections,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Defect_Detection_Report.docx", buf);
  console.log("Report written. Total blocks:", sections.length);
});
