# Roadmap

Alright now that we got all the understanding out of the way, I want to attempt to create a roadmap for this project so I can start executing. I will attempt to create it on my own and then we can redefine what I should actually do and in what order. This will be very broad, later we can get into the details of each pipeline.

1. Exploratory analysis: understand columns, data types, missing values, duplicate/redundant fields... Final feature selection should depend on the task, association rule mining and clustering may need different versions of the dataset. Here we simply find out what we are working with and take notes... maybe.

2. After obtaining an understanding of the data, we need to start defining our project questions. Understand the data, tentatively choose tasks, define what you want to learn from each task. 

3. Choose baseline and find recent methods that best fit to the questions.

4. Define evaluation and experimental design. Choose metrics, sampling strategy if necessary, hyperparameter ranges, random seeds, fairness conditions, and possible ablations.

5. Build preprocessing pipelines appropriate to each task. Share basic cleaning where appropriate, but create task-specific transformations.

6. Implement baseline pipelines first. Establish functioning classical results before introducing recent methods.

7. Implement and compare recent techniques. Use the same evaluation framework for fair comparisons.

8. Run tuning, sensitivity analysis, and meaningful ablations. Record results carefully for the report.

9. Develop the notebook and report throughout the process. Finalize them after experiments, rather than starting them only after all coding is complete.

# Project Description

In this project you will be gaining hands-on experience by tackling a data challenge problem. You will be working individually.

**Programming language:** We will be using Python and Jupyter Notebooks. 

**The data challenge problem:** We will be focusing on the following tabular dataset: 

Data:
https://data.transportation.gov/stories/s/Form-57-Data-Downloads/i5dw-jvsi/

Given this dataset, you are asked to perform two out of the major tasks we have seen in class: (1) classification, (2) clustering, (3) association rule mining, or (4) anomaly detection.

Once you settle on two of the four major tasks, you will have to **run and compare at least two (substantially) different techniques** for performing each chosen task. You should provide an end-to-end solution pipeline around the techniques you chose, that includes **at least two appropriate measures of success/goodness**. 

**Which two techniques should you choose?**
One of the techniques you should choose should be a “classical” one, i.e., a technique that we have seen in class or the textbook. This is considered established for a long time. The second technique you should choose should be a more recently proposed technique that you should search for in relevant literature, and provide a short literature survey that justifies your choice (e.g., a certain technique is highly-cited and well-established in the literature). The point of this is so that you can optimize and compare the performance of “classical” and “cutting-edge” methods (for example, there are recent methods that show how tabular foundation models can perform association rule mining, see: https://arxiv.org/abs/2602.14622). 
 
In case the dataset is too large for your machine, make sure you demonstrate the effectiveness of your solution in smaller representative samples of the data (while providing an average solution to make sure you eliminate any biases introduced by the sampling).

You will not be graded strictly based on the performance of your solution, although you should make your best effort in producing a solution that works on par with the baselines. You will be graded based on a number of points, which include:
- How well you can put in context your solution with respect to existing state-of-the-art 
- How well and how concisely you can describe your solution
- How you were able to efficiently fine-tune your solution
- How thoroughly you compare against known state-of-the-art methods
- Whenever the solution contains different components, is there an ablation study that showcases the effect of each component?
- Whenever the solution contains different hyperparameters, is there thorough sensitivity analysis that shows how different values of those hyperparameters affect the performance?
-	Is the final report descriptive enough so that one can reasonably recreate the proposed solution based on that report?

You may notice that a lot of parameters for this project are loosely defined. This is by design and it is meant to afford you maximum flexibility in tackling this problem. However, you should make sure you set your own parameters/make your own decisions as you are working on this project, and provide justification for them in the final report.

## Final Deliverable

The final deliverable will consist of

1. Report, in PDF format, up to 4 pages (excluding references) in ACM SIGKDD paper submission style  (see https://kdd2024.kdd.org/research-track-call-for-papers/). You should provide a concise description of the problem and relevant work that informed your solution, provide a concise description of your solution pipeline, and show extensive experiments that demonstrate the effectiveness of your solution over standard solutions that are found in the literature (and especially in the original paper that introduced the dataset)
2. Jupyter Notebook that showcases your solution

## Academic Integrity Prompt

- I am working on coursework governed by strict academic integrity rules. For any class project, report, code, analysis, or submitted work, help me learn and produce my own implementation rather than producing submission-ready work for me.

Default to explaining concepts, clarifying requirements, helping me plan an approach, reviewing my reasoning, providing pseudocode, identifying bugs in code I wrote, and suggesting small targeted improvements.

When I ask for code, report text, methodology, experimental design, or other material that could be submitted, clearly warn me when the assistance may be substantial or should be disclosed. Prefer guiding me through building it myself instead of giving a complete final solution. When code examples are useful, keep them limited to illustrative or targeted snippets unless I explicitly request more for learning purposes.

When revising my writing, preserve my wording and ideas as much as possible. Point out issues and suggest changes rather than replacing my work with polished text unless I specifically request a model example.

For project decisions, help me understand alternatives, tradeoffs, evaluation methods, and debugging steps, but do not invent results, experiments, sources, citations, or claims.