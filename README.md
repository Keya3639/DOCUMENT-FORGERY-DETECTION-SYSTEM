# DOCUMENT-FORGERY-DETECTION-SYSTEM
The **Document Forgery Detection System** is a deep learning–based application designed to identify whether a document is ORIGINAL or FORGED using image analysis techniques. Built using Convolutional Neural Networks (CNNs), this system focuses on detecting subtle visual inconsistencies that commonly occur in forged documents, such as alterations, manipulations, or low-quality scans.
The application supports image files (JPG, PNG) as well as PDF documents, processes them into a standardized format, and predicts forgery with a confidence score. It runs as an interactive Streamlit web application, making it simple, intuitive, and suitable for real-world use.

## Overview

This system uses a **trained CNN model** to analyze grayscale document images and learn patterns that distinguish genuine documents from forged ones.
The workflow includes:

- Image preprocessing (resizing, grayscale conversion, contrast enhancement)
- CNN-based feature extraction
- Binary classification (Original vs Forged)
- Confidence-based decision thresholding
- For PDF files, each page is processed independently, allowing page-wise forgery detection.

The model outputs:

- Prediction label (ORIGINAL / FORGED)
- Confidence score
- Raw probability value
- Preprocessed image preview for transparency

## Tools and Technologies Used

- **Python:** Core programming language for development and logic.

- **TensorFlow / Keras:** Used to build, train, and load the CNN model.

- **OpenCV:** Image preprocessing (resizing, grayscale conversion, contrast enhancement).

- **Streamlit:** Interactive web interface for file uploads and result visualization.

- **NumPy:** Numerical operations and image array handling.

- **Pandas:** Maintains upload history and confidence tracking.

- **Matplotlib:** Visualizes confidence trends over multiple uploads.

- **pdf2image & Poppler:** Converts PDF pages into images for analysis.

- **python-docx:** Reads text from DOCX files for preview.

## Why These Tools Were Selected

- CNNs are highly effective for image-based forgery detection.
- TensorFlow/Keras provide flexibility and performance for deep learning models.
- OpenCV enhances document quality for better prediction accuracy.
- Streamlit enables rapid deployment of ML models as web apps.
- pdf2image allows seamless PDF-to-image conversion.
- Pandas and Matplotlib help in tracking and visualizing prediction history.
- Python offers a rich ecosystem for computer vision and ML tasks.

Together, these tools make the system robust, extensible, and user-friendly.

## Features

- Detects forged vs original documents using CNN
- Supports Image (JPG, PNG) and PDF files
- Page-wise analysis for multi-page PDFs
- Contrast enhancement for scanned documents
- Adjustable prediction threshold
- Displays raw model probability
- Shows original vs preprocessed image comparison
- Confidence score visualization
- Downloadable result report
- Upload history tracking
- Simple and clean web interface

## How It Works

- User uploads an image or PDF file
- Image is converted to grayscale and resized to 256×256
- Contrast enhancement is applied
- Image is normalized and passed to the CNN model
- Model outputs probability of forgery
- Threshold-based classification determines final label
- Results are displayed with confidence score
- History and confidence trends are updated
  
## Advantages

- Detects forgery based on learned visual patterns
- Works on scanned and photographed documents
- Page-level detection for PDFs
- No manual feature extraction required
- Easy-to-use web interface
- Can be extended with new document datasets
- Suitable for academic and real-world demonstrations

## Limitations

- Performance depends on training data quality
- Works best on document images, not plain text files
- Cannot explain exact forgery location
- May misclassify very low-quality scans
- Threshold tuning may be required for different datasets
- Not a legal verification tool—acts as a decision support system

## Real-Time Applications

- **Education:** Verify academic certificates and mark sheets
- **Banking & Finance:** Document verification for KYC
- **Recruitment:** Resume and certificate authenticity checks
- **Government Offices:** Identity and record validation
- **Legal Domain:** Preliminary screening of submitted documents
- **Corporate Compliance:** Internal document verification

## Future Enhancements

- Highlight forged regions using explainability techniques
- Support for DOCX and TXT forgery analysis
- Multi-class forgery detection
- Integration with OCR for text-image correlation
- Model explainability (Grad-CAM or attention maps)
- Cloud deployment (AWS / Streamlit Cloud / Hugging Face)
- Export results as PDF reports
- Authentication-based user access

## Conclusion

The Document Forgery Detection System using CNN demonstrates how deep learning can be effectively applied to real-world document verification problems. By leveraging CNN-based visual analysis, the system detects forged documents beyond simple rule-based checks.
With further enhancements and larger datasets, this project can evolve into a powerful tool for academic, professional, and organizational document validation.

## OUTPUT:
<img width="1851" height="805" alt="Image" src="https://github.com/user-attachments/assets/4c53f9a3-98da-42b4-96c2-fe2d9a2ea3b2" />

<img width="1750" height="773" alt="Image" src="https://github.com/user-attachments/assets/ba92e07e-fcfb-4941-8b42-20d082bf281b" />

<img width="1837" height="810" alt="Image" src="https://github.com/user-attachments/assets/3968a76e-c43b-47d8-ad6a-73c71f4505ee" />

<img width="557" height="551" alt="Image" src="https://github.com/user-attachments/assets/b1bfc108-c945-4bb8-b44e-862f320da79e" />

