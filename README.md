
<h1>Classifying_Maqams</h1>

<h2>Classifying_Maqams</h2>

<p><strong>Classifying_Maqams</strong> is a BS.c project aiming to detect and classify Maqams of Quranic recitations using Machine Learning and Deep Learning techniques.</p>

<h3>Project Overview</h3>

<p>The primary objective of this project is to develop a robust and accurate system capable of automatically classifying and detecting the maqam (a system of melodic modes used in Arabic music) of a given audio file. The system leverages deep learning algorithms to extract meaningful audio features, which are then used to predict the corresponding maqam class.</p>

<h3>Contributors</h3>

<ul>
    <li>Faisal Omari, Bs.c student in the Etgar program (University of Haifa).</li>
    <li>Mayas Ghantous, Bs.c student in the Etgar program (University of Haifa).</li>
</ul>

<h3>Installation</h3>

<ol>
    <li>Clone the repository:</li>
</ol>

<pre><code>git clone https://github.com/your-username/Classifying_Maqams.git
cd Classifying_Maqams
</code></pre>

<ol start="2">
    <li>Set up the environment (recommended to use a virtual environment):</li>
</ol>

<pre><code>conda create -n maqam_env python=3.8
conda activate maqam_env
</code></pre>

<ol start="3">
    <li>Install the required packages:</li>
</ol>

<pre><code>pip install -r requirements.txt
</code></pre>

<h3>Usage</h3>

<ol>
    <li><strong>Data Preparation</strong>: Ensure you have the audio dataset organized as specified in the project's data directory.</li>
    <li><strong>Feature Extraction</strong>: The project involves extracting various audio features. Modify and utilize the provided feature extraction scripts according to your dataset.</li>
    <li><strong>Training</strong>: Train the classification model using the provided code. You can experiment with different models, hyperparameters, and training configurations.</li>
    <li><strong>Evaluation</strong>: Evaluate the trained models using the provided testing code. Measure accuracy, ROC curves, and other relevant metrics.</li>
</ol>

<h3>Additional Information</h3>

<ul>
    <li>The <code>data</code> directory should be structured as follows:</li>
</ul>

<pre><code>data/
├── Ajam/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── Bayat/
├── Hijaz/
├── Kurd/
├── Nahawand/
├── Rast/
├── Saba/
└── Seka/
</code></pre>

<p>Detailed instructions for each stage of the project can be found in their respective folders.</p>

<p>For any questions or issues, feel free to open an issue or contact the contributors at <a href="mailto:faisalomari321@gmail.com">faisalomari321@gmail.com</a>.</p>

