**Deep Learning -- Advanced Study Assignment**

**Why Does the Direction of KL Divergence Lead to Different Optimization Behaviors and Modeling Outcomes?**

**A_02 Report** $\mathbf{\bullet}$ **April 2026**

  -------------------------------------------------------------------------------------------------------------------------------------------------------
  **Research Question**   Why does the direction of KL divergence (forward vs. reverse) lead to different optimization behaviors and modeling outcomes?
  ----------------------- -------------------------------------------------------------------------------------------------------------------------------
  **Keywords**            KL divergence, entropy, cross-entropy, mean-seeking, mode-seeking, bimodal distribution, mean, variance

  **Tools**               PyTorch, NumPy, Matplotlib, AI tools (GPTs), Wikipedia, Articles

  **Dataset**             Synthetic bimodal Gaussian mixture (toy experiment)
  -------------------------------------------------------------------------------------------------------------------------------------------------------

## Abstract

The Kullback--Leibler divergence (KL divergence) is a fundamental quantity in machine learning, deep learning, and information theory, widely used to measure the difference between probability distributions. Despite its central role in probabilistic modelling, its inherent asymmetry is often overlooked in practical applications. This asymmetry arises from the direction of evaluation, specifically from which distribution defines the expectation: forward KL divergence D~KL~(P∥Q) evaluates discrepancies under the target distribution, while reverse KL divergence D~KL~(Q∥P) evaluates them under the model distribution, leading to fundamentally different optimization behaviours.

This study investigates how the direction of KL divergence affects optimization dynamics and modeling outcomes. A theoretical analysis based on gradient behavior is combined with controlled experiments in which a unimodal Gaussian model is trained to approximate a bimodal target distribution. The experiments systematically examine the effects of initialization, learning rate, and mode separation.

The results demonstrate that forward KL divergence consistently produces mode-covering behaviour, characterized by higher entropy, greater stability, and robustness to initialization. In contrast, reverse KL divergence exhibits mode-seeking behavior, leading to lower entropy, sensitivity to initialization, and convergence toward local modes. These differences become more pronounced as the complexity and separation of the target distribution increase.

The findings highlight that the direction of KL divergence is not a minor implementation detail but a critical design choice that governs whether a model prioritizes distributional coverage or concentrated approximation.

## 1. Introduction and Motivation

The comparison of probability distributions is a central problem in machine learning, deep learning, statistics, and information theory. Among the most widely used measures for quantifying divergence between distributions is the Kullback--Leibler divergence (KL divergence), which plays a fundamental role in learning algorithms ranging from maximum likelihood estimation to variational inference and deep generative modelling.

Despite its broad applicability, KL divergence possesses an important but often underemphasized property: **asymmetry**. Specifically, the divergence between two probability distributions *P(x)* and *Q(x)* depends on the direction of evaluation, such that:

*D~KL~ (P ∥ Q) ≠ D~KL~ (Q ∥ P)*

This asymmetry implies that the choice of direction is not merely a mathematical formality, but a decision that can significantly influence learning behaviour, optimization dynamics, and model bias^1^.

In modern deep learning systems, KL divergence appears in multiple contexts, including variational autoencoders, reinforcement learning policy updates, and probabilistic inference frameworks such as Variational inference. However, the practical consequences of selecting either forward or reverse KL divergence are often treated implicitly rather than explicitly analyzed.

In this study I would like to systematically investigate the influence of KL divergence direction on optimization outcomes and learned representations. Using controlled synthetic experiments implemented in **PyTorch**, then I will analyze how forward and reverse KL divergences behave under varying initialization conditions, learning rates, and target distribution complexities.

The primary objective of this work is to answer the following question:

*"How does the direction of KL divergence influence the optimization behavior, and Modeling Outcomes?"*

As it is hypothesized that forward KL divergence encourages mode-covering behavior and higher uncertainty, whereas reverse KL divergence promotes mode-seeking behavior and sharper, more concentrated distributions.

*Why the question is interesting?*

In my first report, I found entropy and cross-entropy very closely linked to the KL divergence. But to fully develop my intuition, I would like to cover all other aspects of KL divergence, why it is related to entropy, cross-entropy and why it has two different directions.

*How it is connected to the lecture material?*

In our lecture material, many aspects are already covered, and I could follow the direction of the materials covered. However, this assignment is purely given to us to develop the critical thinking and the flow of Deep Learning, where we should understand it is not about listening to the lecture but taking time for deep research and making more self-disciplined study.

*And Why you chose it?*

As I already stated that the investigation of this topic is more about developing my intuition and clean understanding of Deep Learning, where I can connect and clearly understand the mechanism of KL divergence, while it gives me time to read and develop the theory behind, get well use of available resources.

One example where in this slide, I would like to be quietened myself how and why we need such an approach, and where such intuition should be revoked, or how we know when to use forward or reverse, and more other questions lead me to make this report.

![](kl_md_assets/media/image1.png){width="6.268055555555556in" height="3.423611111111111in"}

Now let's start to deep dive into the theoretical part and then to experimental.

## 2. Background and Theoretical Framework

### **2.1 I**nformation-Theoretic Foundations^2^

### 

KL divergence arises naturally from information theory. The entropy of a distribution P(x) measures the optimal coding length:

![](kl_md_assets/media/image2.png){width="2.166126421697288in" height="0.5386964129483814in"}

When we approximate P(x) using another distribution Q(x), the expected coding cost becomes the cross-entropy:

![](kl_md_assets/media/image3.png){width="2.5941076115485564in" height="0.6449442257217848in"}

The KL divergence is defined as the difference between these two quantities:

![](kl_md_assets/media/image4.png){width="2.794350393700787in" height="0.45005139982502185in"}

Thus, KL divergence quantifies the additional cost (extra coding length) added when using Q to approximate P.

### 2.2 KL Divergence Definition^2^

### 

The **Kullback--Leibler (KL) divergence** is a measure of distance between two probability distributions P(x) and Q(x), defined as:

![](kl_md_assets/media/image5.png){width="2.9262128171478565in" height="0.6303105861767279in"}

Where:

-   P(x): true data distribution

-   Q(x): model distribution

It is important to note that KL divergence is not a true distance metric, as it is not symmetric and does not satisfy the triangle inequality. Instead, it should be interpreted as a measure of discrepancy between two distributions.

### 2.3 Forward vs Reverse KL

### 

### 2.3.1 Forward KL Formulation (P \|\| Q): 

Forward KL computes the expectation over the true data distribution P.

The formula is mathematically represented as:

![](kl_md_assets/media/image6.png){width="5.801268591426072in" height="0.8383519247594051in"}

![](kl_md_assets/media/image7.png){width="6.768408792650918in" height="1.599844706911636in"}

### 2.3.2 Reverse KL Formulation (Q \|\| P):

Reverse KL computes the expectation over the model\'s predicted distribution Q.

The formula is:

![](kl_md_assets/media/image8.png){width="6.268055555555556in" height="0.6820231846019248in"}

![](kl_md_assets/media/image9.png){width="6.268055555555556in" height="3.6215277777777777in"}

![](kl_md_assets/media/image11.png){width="6.268055555555556in" height="3.452777777777778in"}

## 2.4 KL divergence is not symmetric ^2-6^

A key property of the Kullback--Leibler (KL) divergence is its asymmetry:

D~KL~​(P∣∣Q)$\neq$D~KL~​(Q∣∣P)

As we know already KL divergence formulation lets go step by step to the explanation of asymmetry:

### 2**.4.1** Formal Explanation of the Source of Asymmetry

At first glance, the two divergences

![](kl_md_assets/media/image12.png){width="6.268055555555556in" height="0.7180555555555556in"}

appear structurally similar. However, exchanging P and Q induces **two simultaneous and non-equivalent changes**:

1\. Change in the Weighting Measure

The KL divergence is an **expectation with respect to a specific distribution**:

![](kl_md_assets/media/image13.png){width="3.2252799650043746in" height="1.5084645669291339in"}

Thus, exchanging P and Q changes the **underlying measure of integration (or summation)**:

-   In D~KL~(P∥Q), outcomes are weighted according to their probability under P

-   In D~KL~(Q∥P), outcomes are weighted according to their probability under Q

Since P≠Q in general, these expectations are taken over **different probability spaces**, which already prevents symmetry.

2\. Change in the Logarithmic Term

Simultaneously, the argument of the logarithm is inverted:

![](kl_md_assets/media/image14.png){width="4.47538823272091in" height="0.8750754593175853in"}

While this introduces a sign change at the level of the integrand, it does **not** produce a simple negation of the divergence because the weighting distribution also changes.

![](kl_md_assets/media/image15.png){width="5.659194006999125in" height="3.3111242344706913in"}

This asymmetry can be understood intuitively by interpreting KL divergence as a measure of **"surprise"**. Specifically, it quantifies how surprising it is to observe samples from one distribution when they are evaluated under another distribution.

More formally, D~KL~(P∥Q) measures how well the distribution Q explains samples drawn from P. Reversing the arguments changes the sampling process, and therefore changes the notion of "surprise." As a result, the two quantities are fundamentally different.

### 2.3.1 Conceptual intuition.

KL divergence measures \"surprise.\" Specifically, it asks: if our sample points from one distribution (P), how surprised would we be if those points actually modelled a different distribution (Q)? Because sampling from P to get Q is a different process than sampling from Q to get P, the resulting \"surprise\" values are different.

I would like to give some illustration examples bellow.

![](kl_md_assets/media/image16.png){width="5.5465277777777775in" height="3.4940791776027997in"}

![](kl_md_assets/media/image17.png){width="6.268055555555556in" height="2.2082064741907264in"}

**Simple Illustration:**

![](kl_md_assets/media/image18.png){width="6.4457655293088365in" height="3.003170384951881in"}

**The Salad vs. Tomato Analogy**

**P is a salad** and **Q is a bag of tomatoes**.

-   **KL (P, Q):** If our sample from a salad, we aren\'t very surprised to pull out a tomato. This results in a small KL divergence.

-   **KL (Q, P):** If our sample from a bag of tomatoes, we would be extremely surprised to pull out a full salad (lettuce, onions, dressing). This results in a large KL divergence.

KL Divergence is not a geometric distance; it is a measure of how easily one distribution can represent or generate another. Because the direction of this transformation matters, the symmetry does not hold.

## 2.5. Strengths and Limitations

The choice between forward and reverse KL divergence is rarely a matter of one being \"better\" than the other; rather, it is a choice of which type of error the practitioner is willing to tolerate.

**2.5.1. Forward KL (D~KL~ (P \|\| Q))**

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Strengths**                                                                                                                               **Limitations**
  ------------------------------------------------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Global Coverage:** Guarantees that every mode of the target distribution is represented. It is \"all-encompassing.\"                      **Over-generalization:** Often places high probability mass in \"empty\" regions (valleys) where the true probability is zero.

  **Initialization Robustness:** The optimization landscape is generally smoother, making it less likely to get stuck in poor local minima.   **Blurry Approximations:** In high-dimensional spaces (like images), this leads to \"blurry\" results because the model tries to average all possible outcome

  **Entropy Maximization:** Naturally captures the full variance and uncertainty of the data.                                                 **Computationally Expensive:** Requires sampling from the true distribution P, which is often unknown or represented only by a finite dataset.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**2.5.2. Reverse KL (D~KL~ (Q \|\| P))**

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Strengths**                                                                                                                                                                   **Limitations**
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **High Precision:** The model produces very \"realistic\" samples that are guaranteed to stay within the high-probability regions of the target.                                **Mode Collapse:** Completely ignores parts of the data. If the target has 10 modes, the model might only learn one and \"forget\" the other nine.

  **Computational Efficiency:** Does not require samples from the true \$P\$; it only needs to be able to evaluate the density P(x), making it ideal for Variational Inference.   **Initialization Sensitivity:** Highly dependent on where the model starts. A poor initialization can lead to a \"dead\" model that never finds the target mass.

  **Sharp Representations:** Produces low-entropy, concentrated distributions that are useful for decision-making and optimization.                                               **Underestimates Uncertainty:** Often leads to \"overconfidence,\" where the model is very certain about a narrow range of values while ignoring others.
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

We can also frame this using the **\"Zero-Avoiding\" vs. \"Zero-Forcing\"** terminology:

1.  **Forward KL is \"Zero-Avoiding\":** It avoids assigning a probability of zero to any region where the target is non-zero *(Q(x) ≠ 0 if P(x) \> 0).*

> **Strength:** We don\'t miss anything.
>
> **Limitation:** We get a lot of \"noise\" or \"blur.\"

2.  **Reverse KL is \"Zero-Forcing\":** It forces the model to be zero wherever the target is zero *(Q(x) = 0 if P(x) = 0).*

> **Strength:** We get very \"clean\" and \"accurate\" samples.
>
> **Limitation:** We lose the \"big picture\" of the data.

### 2.6 Gradient Behavior Comparison and Training Signal:

The fundamental difference between the two loss functions lies in how they penalize errors during optimization.

-   **Forward KL is \"Zero-Avoiding\" (Mean-Seeking):**

> Looking at the ratio $\frac{P_{(x)}}{Q_{(x)}}$. If the true data P(x) \> 0 but the model predicts Q(x) $\rightarrow$ 0, the ratio approaches infinity, resulting in a massive penalty. Therefore, the model Q is forced to stretch out and cover *every* region where true data exists. If the true data has two distinct peaks (bimodal) and our model can only create one peak (unimodal), the model will place its broad peak right in the middle to cover both. This leads to safe, but \"blurry\" or averaged predictions.

-   **Reverse KL is \"Zero-Forcing\" (Mode-Seeking):**

> Looking at the ratio $\frac{Q_{(x)}}{P_{(x)}}$ in the Reverse KL formula. If the model predicts Q(x) \> 0 in a region where the true data P(x) $\rightarrow$ 0, the penalty explodes. To avoid this, the model Q must be exactly zero wherever P is zero. Faced with bimodal true data, a unimodal model cannot stretch across the empty space between the peaks. Instead, it will collapse and tightly fit onto just *one* of the peaks, completely ignoring the other. This causes \"mode collapse.\"

### 2.7 Practical Implications

The theoretical \"mode-covering\" vs. \"mode-seeking\" behavior translates into specific design outcomes in modern AI systems.

### 2.7.1 Generative Modeling: Diversity vs. Fidelity

In generative AI, there is a constant tension between **diversity** (showing all possible variations) and **fidelity** (how realistic each sample looks).

**Forward KL (Likelihood-based models):** Models like Autoregressive models or Variational Autoencoders (VAEs) that prioritize likelihood tend to be \"diverse.\" They will try to cover every image in the training set. However, because they are \"mode-covering,\" they often produce blurry results where the model averages out different possibilities.

**Reverse KL (Adversarial/Variational Inference):** Many GAN-based approaches or specific VI frameworks behave like reverse KL. They produce incredibly sharp, high-fidelity images. The trade-off is **\"Mode Collapse\"**---the model might only learn to draw three types of perfect faces and completely \"forget\" that other ethnicities or lighting conditions exist.

### 2.7.2 Variational Inference (VI): The \"Overconfidence\" Problem

### 

In Bayesian deep learning, we use Reverse KL because we usually can't sample from the true posterior P.

**The Implication:** Because Reverse KL is \"zero-forcing,\" the resulting model is almost always **under-dispersed**.

**The Risk:** The model will be \"overconfident.\" It might give a prediction with 99% certainty because it has collapsed onto one local mode of the truth, completely ignoring other valid interpretations of the data. This is dangerous in high-stakes fields like **autonomous driving** or **medical prognosis**, where knowing what you *don\'t* know is critical.

### 2.7.4 Reinforcement Learning (RL): Exploration vs. Exploitation

### 

In RL algorithms like **Proximal Policy Optimization (PPO)** or **Soft Actor-Critic (SAC)**, KL divergence is used to ensure the \"new\" policy doesn\'t stray too far from the \"old\" policy.

**Forward KL Implication:** Encourages the agent to keep its options open. It prevents the policy from prematurely committing to one path, maintaining better **exploration**.

**Reverse KL Implication:** Encourages the agent to refine a specific successful strategy. This leads to faster **exploitation** and convergence but risks the agent getting stuck in a sub-optimal \"local\" strategy (a local mode) and never finding the global best.

## 3. Experimental Verification

### 3.1 Experimental Setup

### 

**Classic experiment comparing Forward KL (Mean-Seeking) and Reverse KL (Mode-Seeking) by fitting a single Gaussian Q(x) to a bimodal Gaussian mixture P(x).**

To investigate why the direction of Kullback--Leibler (KL) divergence leads to different optimization behaviors, we construct a controlled synthetic experiment. The key idea is to intentionally create a **model mismatch scenario**, where the model is not expressive enough to perfectly represent the true data distribution. This forces the optimization process to reveal its inherent bias.

Specifically, we approximate a **bimodal target distribution** using a **unimodal model**. Because the model cannot capture both modes simultaneously, the optimization objective must decide how to allocate probability mass. This provides a clear setting to observe how forward and reverse KL divergence behave differently.

**Conceptual Clarification of the Problem**

To avoid ambiguity, we explicitly define the key components of the problem:

### **3.1.1 Target Distribution** P(x)

### 

The true data-generating distribution P(x)P(x)P(x) is defined as a symmetric mixture of two Gaussian components:

![](kl_md_assets/media/image19.png){width="5.067106299212599in" height="0.5000437445319335in"}

• N (x∣μ,σ^2^) denotes a Gaussian density

• the distribution has **two symmetric modes** at x=−μ and x=μ

• Variance of each component is fixed to 1

• the parameter μ\>0 controls **mode separation**

The equal weighting (0.5, 0.5) ensures perfect symmetry, meaning neither mode is preferred. Fixing the variance to 1 isolates μ as the only factor controlling task difficulty

Interpretation:

-   Small μ: strong overlap → easier approximation

-   Large μ: well-separated modes → harder approximation

**Important**

**1. Why the 0.5 (The Mixing Weights)**

The 0.5 values represent the **weight** or **probability** of each of the two Gaussian distributions in the mixture.

**Symmetry:** By setting both weights to 0.5, it ensures the two \"peaks\" (modes) of the distribution are exactly the same height. Neither side is heavier or more likely than the other.

**Law of Probability:** In a valid probability distribution, the total probability must sum to 1. Since there are two distributions and we want them to be equal, we divide that probability in half (0.5 + 0.5 = 1). This means a data point has a 50% chance of coming from the left distribution and a 50% chance of coming from the right one.

**2. Why the 1 (The Variance)**

The 1 inside the normal distribution notation ![](kl_md_assets/media/image20.png){width="0.5460684601924759in" height="0.16132327209098862in"} represents the **variance** (σ^2^).

**Constant Width:** Setting the variance to 1 fixes the \"width\" or \"fatness\" of both bell curves to a standard size.

**Isolating the Difficulty:** Because the widths are locked at a constant 1, the *only* thing that determines how much the two distributions overlap is the distance between them (controlled by μ). If the variance were allowed to change, the overlap would depend on a messy combination of both distance (μ) and width (σ^2^). Keeping the variance at 1 isolates μ as the single \"dial\" you can turn to make the approximation harder or easier.

### 3.1.2 Model Distribution Q(x)

The approximating model Q(x) is defined as:

![](kl_md_assets/media/image21.png){width="3.341956474190726in" height="0.7250623359580053in"}

where both mean μ~q~​ and variance σ^2^~q~​ are learnable parameters

Because Q(x) is unimodal, it cannot represent both modes of P(x), forcing the optimization process to choose how to approximate the target distribution.

### 3.1.3 KL Divergence Objectives

We compare two optimization objectives:

**Forward KL**

![](kl_md_assets/media/image22.png){width="3.133604549431321in" height="0.43337051618547684in"}

Since logP(x) does not depend on the model parameters, minimizing forward KL reduces to:

![](kl_md_assets/media/image23.png){width="1.766819772528434in" height="0.4417049431321085in"}

This corresponds to maximizing likelihood under samples drawn from P(x).

**Reverse KL**

![](kl_md_assets/media/image24.png){width="3.1086023622047243in" height="0.43337051618547684in"}

In this case, the expectation is taken with respect to the model distribution Q(x), making the optimization dependent on where the model currently assigns probability mass.

### **3.1.4 Implementation**

The experiment is implemented in PyTorch. Sampling from the target distribution is performed explicitly by first selecting one of the two Gaussian components with equal probability and then drawing from that component.

**Implementation of P(x)**

![](kl_md_assets/media/image25.png){width="6.268055555555556in" height="1.4638888888888888in"}

**Explanation of the Code**

-   A Bernoulli random variable selects between the two Gaussian components.

-   If the sampled value is 0 → mean = −μ

-   If the sampled value is 1 → mean = +μ

-   A standard normal sample is then shifted by the selected mean.

This process exactly matches the mathematical definition of the mixture distribution.

**Log-Density Implementation**

![](kl_md_assets/media/image26.png){width="6.268055555555556in" height="2.0965277777777778in"}

These functions compute the log-probability under the model and the target distribution.

**Optimization Procedure**

The model parameters μ ​ and σ~q~​ are optimized using gradient descent.

![](kl_md_assets/media/image27.png){width="6.268055555555556in" height="4.732638888888889in"}

Learning rates and other hyper parameters will be adjusted during the experiments (Appendix)

## **3.2 Experimental Results**

We evaluate the model under:

-   Mode separation: μ=1,2,4

-   Initialization: μ~q~= −3,0,3

-   KL direction: Forward vs Reverse

![](kl_md_assets/media/image28.png){width="6.268055555555556in" height="1.9159722222222222in"}

### Observations

-   **Forward KL**:

    -   mean ≈ 0

    -   variance → very large

-   **Reverse KL**:

    -   mean ≈ one mode

    -   variance → small

### Interpretation

This demonstrates that:

-   Forward KL produces **mode-covering behavior**

-   Reverse KL produces **mode-seeking behavior**

### 3.2.1 Quantitative Comparison

To summarize the results, we compute the average variance and convergence behavior across settings.

![](kl_md_assets/media/image29.png){width="6.730444006999125in" height="1.0531189851268592in"}

### 3.2.2 Effect of Mode Separation

![](kl_md_assets/media/image30.png){width="6.268055555555556in" height="2.79375in"}

![](kl_md_assets/media/image31.png){width="6.268055555555556in" height="1.586111111111111in"}

variance stays \~1

always picks ONE mode

What's really happening (correct explanation)

Case 1 --- Large separation (μ = 4)

Modes are far apart

Reverse KL picks one mode cleanly

So Q(x) ≈ one Gaussian component

✔ Result:

→ variance ≈ 1 (correct)

Case 2 --- Medium separation (μ = 2)

Modes overlap slightly

Q(x) still picks one mode BUT:

it spreads a bit to reduce loss

✔ Result:

→ variance increases (\~4)

Case 3 --- Small separation (μ = 1)

Modes overlap heavily

The mixture looks almost unimodal

✔ Result:

→ reverse KL behaves more like forward KL

→ variance increases (\~2)

### **3.2.3 Initialization Sensitivity**

![](kl_md_assets/media/image32.png){width="6.268055555555556in" height="1.8791666666666667in"}

![](kl_md_assets/media/image33.png){width="5.917492344706911in" height="1.7502438757655292in"}

-   Different initializations lead to different final modes

### Interpretation

Reverse KL has multiple local optima and is sensitive to initialization.

Forward KL does not show this behaviour.

### 3.2.4 Quantitative Analysis

![](kl_md_assets/media/image34.png){width="5.459095581802274in" height="2.344076990376203in"}

This provides numerical evidence supporting the qualitative observations.

### **3.2.4 Convergence Analysis**

Convergence speed differs between the two objectives:

-   Forward KL shows **consistent convergence** across all settings

-   Reverse KL shows **variable convergence**, sometimes converging faster but less reliably

This suggests that reverse KL creates a more complex optimization landscape.

### **3.2.5 Visualization**

![](kl_md_assets/media/image35.png){width="6.268055555555556in" height="3.067361111111111in"}

**Figure 1.**

The qualitative behavior is illustrated in Figure 1:

-   Forward KL produces a **wide distribution** covering both modes

-   Reverse KL produces a **narrow distribution** focused on a single mode

This visually confirms the quantitative findings.

## **3.3 Explanation of Observed Behavior (Answer to Research Question)**

The experimental results can be directly explained by the **asymmetry of KL divergence**, specifically the difference in expectation.

### **Expectation Difference**

-   Forward KL:

![](kl_md_assets/media/image36.png){width="0.6667246281714786in" height="0.35003062117235345in"}

→ evaluates all regions where the true distribution has probability

-   Reverse KL:

![](kl_md_assets/media/image37.png){width="0.744329615048119in" height="0.46157917760279965in"}

→ evaluates only regions where the model already assigns probability

### **Penalty Asymmetry**

This leads to fundamentally different optimization pressures:

-   **Forward KL penalizes missing probability mass**

    -   encourages covering all modes

    -   leads to high variance

-   **Reverse KL penalizes placing mass in low-density regions**

    -   encourages focusing on one mode

    -   leads to low variance

### **Final Interpretation**

The direction of KL divergence fundamentally changes the optimization objective by altering:

-   which regions of the distribution are emphasized

-   how errors are penalized

As a result:

-   Forward KL produces **mode-covering solutions**

-   Reverse KL produces **mode-seeking solutions**

### **3.2.1 Empirical Observations**

### 

Forward KL (Mode-Covering Behavior)

Forward KL consistently produces solutions with:

-   mean values near zero (center between modes),

-   large variances covering both modes.

This indicates that the model attempts to **capture all regions of high probability mass**, even at the cost of reduced precision.

Reverse KL (Mode-Seeking Behavior)

Reverse KL results in:

-   convergence to a single mode,

-   small variance concentrated around that mode,

-   strong dependence on initialization.

This demonstrates that the model prioritizes **high-density regions** and ignores others.

**Effect of Mode Separation**

As μ increases:

-   Forward KL further increases variance to maintain coverage,

-   Reverse KL becomes more confident in selecting a single mode.

**Convergence Behavior**

-   Forward KL converges consistently across initializations,

-   Reverse KL shows sensitivity to initialization, indicating multiple local optima.

![](kl_md_assets/media/image38.png){width="6.139376640419948in" height="3.785544619422572in"}

**Figure 2.**

**Figure 2** presents the learned variance of the model distribution as a function of mode separation μ for both forward and reverse KL divergence.

For small μ (μ = 1), both methods produce similar variance values, as the two modes of the target distribution significantly overlap and resemble a unimodal distribution.

As μ increases, the behavior diverges. Forward KL shows a rapid increase in variance, reaching very large values when μ = 4. This indicates that the model attempts to cover both modes by spreading its probability mass across the entire space.

In contrast, reverse KL initially increases variance slightly for moderate separation (μ = 2), but then sharply decreases it for larger μ. When the modes are clearly separated, the model collapses onto a single mode with low variance, avoiding regions of low probability density.

This demonstrates that forward KL promotes mode-covering behavior, while reverse KL promotes mode-seeking behavior, with the transition depending on the degree of separation between modes.

## 

**4. Interpretation**

Through this study, it became clear that the theoretical understanding of KL divergence is both broader and more intuitive than initially expected. From a practical perspective, the choice between forward and reverse KL divergence depends on how we want the model to treat the data distribution.

Specifically, forward KL divergence is appropriate when the goal is to ensure that the model covers all regions where data exists, even if this leads to overestimation or increased variance. In contrast, reverse KL divergence is more suitable when the objective is to focus on the most probable regions of the data, potentially ignoring less significant areas.

Therefore, the direction of KL divergence reflects a fundamental modeling decision: whether to prioritize comprehensive coverage of the data distribution or to concentrate on its most dominant modes.

During my investigation the 3 unexpected situations I have noticed.

First was that reverse KL does not always produce low-variance, mode-seeking solutions. When the modes are not clearly separated (e.g., μ = 2), the model increases its variance instead of collapsing to a single mode. This indicates that reverse KL behavior depends on the structure of the target distribution rather than following a strict rule.

Second, reverse KL shows strong sensitivity to initialization, converging to different modes depending on the starting point. This suggests the presence of multiple local optima and a more complex optimization landscape compared to forward KL.

Finally, the use of a unimodal Gaussian model limits the expressiveness of the approximation. While this simplification helps isolate the effect of KL asymmetry, it may exaggerate differences that would be less pronounced in more flexible models.

## 5. Conclusion

This study investigated how the direction of Kullback--Leibler (KL) divergence influences optimization behavior and modeling outcomes when approximating a bimodal distribution with a unimodal model.

The results clearly demonstrate that the direction of KL divergence leads to fundamentally different solutions due to the asymmetry in how expectations are computed. Forward KL divergence, which evaluates discrepancies under the target distribution, forces the model to account for all regions where data exists. As a result, it produces **mode-covering behavior**, characterized by a central mean and significantly increased variance as the difficulty of the problem increases.

In contrast, reverse KL divergence evaluates discrepancies under the model distribution, meaning that it only considers regions where the model already assigns probability mass. This leads to **mode-seeking behavior**, where the model concentrates on a single mode and avoids low-probability regions. Additionally, reverse KL was found to be sensitive to initialization and capable of converging to multiple local optima.

The experiments further revealed that this behavior is influenced by the structure of the target distribution. When the modes are clearly separated, reverse KL strongly collapses to a single mode, whereas for overlapping modes, it behaves more like a variance-adjusting approximation.

Overall, this study shows that the choice between forward and reverse KL divergence is not merely a technical detail, but a fundamental modeling decision. It determines whether the model prioritizes comprehensive coverage of the data distribution or focuses on its most probable regions, and therefore has a direct impact on the resulting learned representation.

## References

**1. Elements of Information Theory Thomas M. Cover, Joy A. Thomas. 1991**

**2. Information Theory, Inference, and Learning Algorithms, David J.C. MacKay, 2003**

**3. Bishop-Pattern-Recognition-and-Machine-Learning-2006**

**4. Forward and Reverse KL Divergence ([Rohan Tangri](https://towardsdatascience.com/author/rohan-tangri/), 2021**)

**5. Jaynes, E. T. (1957). [\"Information theory and statistical mechanics\"](http://bayes.wustl.edu/etj/articles/theory.1.pdf) Physical Review. 106 **

**6. [Kullback, S.](https://en.wikipedia.org/wiki/Solomon_Kullback); [Leibler, R.A.](https://en.wikipedia.org/wiki/Richard_Leibler) (1951). [\"On information and sufficiency\"](https://doi.org/10.1214%2Faoms%2F1177729694). [Annals of Mathematical Statistics](https://en.wikipedia.org/wiki/Annals_of_Mathematical_Statistics). 22**  

**7. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians." Journal of the American Statistical Association, 112(518), 859--877.**

**8. Minka, T. (2005). "Divergence Measures and Message Passing." Microsoft Research Technical Report MSR-TR-2005-173.**

**9. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes." International Conference on Learning Representations (ICLR).**

**10. Goodfellow, I., et al. (2014). "Generative Adversarial Nets." Advances in Neural Information Processing Systems (NeurIPS).**

![](kl_md_assets/media/image39.png){width="6.268055555555556in" height="4.763194444444444in"}

![](kl_md_assets/media/image40.png){width="6.268055555555556in" height="5.995833333333334in"}

![](kl_md_assets/media/image41.png){width="6.268055555555556in" height="5.266666666666667in"} ![](kl_md_assets/media/image42.png){width="6.268055555555556in" height="5.63125in"}

## ![](kl_md_assets/media/image43.png){width="6.268055555555556in" height="6.125694444444444in"}

## ![](kl_md_assets/media/image44.png){width="6.268055555555556in" height="6.7659722222222225in"}

## ![](kl_md_assets/media/image45.png){width="6.268055555555556in" height="5.914583333333334in"}

## ![](kl_md_assets/media/image46.png){width="6.268055555555556in" height="4.361111111111111in"}

## ![](kl_md_assets/media/image47.png){width="6.268055555555556in" height="2.6333333333333333in"}

![](kl_md_assets/media/image48.png){width="6.268055555555556in" height="5.016666666666667in"}

![https://i.sstatic.net/KgQmM.png](kl_md_assets/media/image49.png){width="6.268055555555556in" height="6.3158661417322834in"}

![](kl_md_assets/media/image50.png){width="6.268055555555556in" height="3.9868055555555557in"}

### 

### 
