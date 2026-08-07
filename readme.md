# Small LUMI ML Demo

A short, hands-on introduction to running deep-learning workloads on the
[LUMI](https://www.lumi.csc.fi) supercomputer. You will work through two parts:

| Part | What you will do | How you run it |
| --- | --- | --- |
| **[Part 1](#part-1-web-interface-exercise)** | Explore the LUMI web interface and run a PyTorch notebook | Interactive Jupyter session |
| **[Part 2](#part-2-slurm-job-exercise)** | Train image classifiers on two datasets | Batch jobs via Slurm |

> [!NOTE]
> Throughout this demo the course project is `project_465002757`. Wherever you
> see `$USER` in a command, it is automatically replaced by your own username,
> so you can copy the commands as-is.

---

## Part 1: Web Interface Exercise

*Introduction to notebooks and PyTorch fundamentals.*

In this part you will get your first experience with the LUMI web interface:
navigating files, setting up your own copy of the exercises, and running a
notebook in an interactive Jupyter session. Working in your own subdirectory
keeps your files separate from the other course participants.

### Step 1: Set up your copy of the exercises

1. Log in to the LUMI web interface: <https://www.lumi.csc.fi>
2. Open the [login node shell app](https://www.lumi.csc.fi/pun/sys/shell/ssh/default).
3. Create your own subdirectory (named after your username) in both the
   project and scratch areas:

   ```bash
   mkdir -p /project/project_465002757/$USER
   mkdir -p /scratch/project_465002757/$USER
   ```

4. Clone the [exercise repository](https://github.com/marlon-tobaben/small-lumi-ml-demo)
   into your project folder:

   ```bash
   git clone https://github.com/marlon-tobaben/small-lumi-ml-demo.git /project/project_465002757/$USER
   ```

### Step 2: Start an interactive Jupyter session

Open the **Jupyter** app in the LUMI web interface and launch it with the
settings below.

> [!WARNING]
> Use the **Jupyter** app and *not* "Jupyter for Courses".

| Setting | Value |
| --- | --- |
| Project | `project_465002757 (LUST Training ...)` |
| Reservation | None |
| Partition | `small-g` |
| Number of CPU cores | `7` |
| Memory (GB) | `16` |
| Time | `0:30:00` |
| Working directory | `/project/$PROJECT` |
| Python | `lumi-multitorch (PyTorch, LUMI AI Factory)` |
| Module version | default |
| Enable virtual environment | **Do not** select this |

Press **Launch**, wait for the session to start, then press
**Connect to Jupyter**.

> [!NOTE]
> Jupyter opens in a new tab. Your interactive job keeps running even if you
> close the tab, so you can always reconnect via the **My Interactive Session**
> page. When you are done, explicitly **Cancel** the job from there. Otherwise
> it keeps consuming your allocation until the time limit is reached.

### Step 3: Run the notebook

Open [`01-pytorch-mnist-mlp.ipynb`](01-pytorch-mnist-mlp.ipynb) and follow along. The notebook trains a **multi-layer perceptron (MLP)** to classify handwritten digits from the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset using PyTorch.

> [!TIP]
> If parts of the notebook disappear as you scroll, this is a
> [known JupyterLab issue](https://github.com/jupyterlab/jupyterlab/issues/17023).
> Workaround: Set the windowing mode to "defer":
> **Settings → Settings Editor → search "windowing mode" → set to `defer`**
> (instead of the default `full`).

---

## Part 2: Slurm Job Exercise

In this part you train image classifiers as **batch jobs** on two datasets, one
per task.

### The datasets

**Dogs vs. cats** (`dvc`) used in [Task 1](#task-1--dogs-vs-cats). 2000
training images, each showing either a cat or a dog.

<img src="https://raw.githubusercontent.com/csc-training/intro-to-dl/e2bdda765b52e1028cd856f49d9b07ef02f7ff8d/day2/imgs/dvc.png" alt="Sample dogs-vs-cats images" width="500">

The second dataset, **German traffic signs** (`gtsrb`), is introduced in
[Task 2](#task-2--german-traffic-signs).

> [!NOTE]
> Both datasets are already staged for the course under
> `/scratch/project_465002757/data` (in the `dogs-vs-cats/train-2000` and
> `gtsrb/train-5535` subfolders). `run.sh` sets the `DATADIR` environment
> variable to this location automatically, so you do not need to download
> anything. If a run stops with `Please set DATADIR environment variable!` or a
> "directory not found" error, ask the course organizers to confirm the data is
> present at that path.

### Getting started

1. Log in to the LUMI web interface: <https://www.lumi.csc.fi>
2. Open the [login node shell app](https://www.lumi.csc.fi/pun/sys/shell/ssh/default).
3. Navigate to your exercise directory:

   ```bash
   cd /project/project_465002757/$USER
   ```

### Task 1: Dogs vs. cats

Train, evaluate, and report the test accuracy with **two different approaches**:

| Approach | Script |
| --- | --- |
| CNN trained from scratch | [`pytorch_dvc_cnn_simple.py`](pytorch_dvc_cnn_simple.py) |
| Pre-trained CNN (VGG16) + fine-tuning | [`pytorch_dvc_cnn_pretrained.py`](pytorch_dvc_cnn_pretrained.py) |

Submit a training job with `sbatch`, passing the script you want to run:

```bash
sbatch run.sh pytorch_dvc_cnn_simple.py
```

**Monitoring your job**

- Check the status of your runs:

  ```bash
  squeue --me
  ```

- The output appears in a file named `slurm-RUN_ID.out`, where `RUN_ID` is the
  Slurm batch job id. Show its last ten lines with:

  ```bash
  tail slurm-RUN_ID.out
  ```

- Follow the output live with `tail -f slurm-RUN_ID.out` (press `Ctrl-C` to stop
  following).

**Reading the results**

After training, the script evaluates on the test set. Look near the end of the
output log for a line starting with `Testing`. It reports the accuracy
(percentage of correctly classified images).

> [!NOTE]
> The pre-trained model prints **two** results: once after pre-training, and
> again after fine-tuning.

Once both runs finish, consider:

- Which model gave the best test-set result?
- Does fine-tuning improve the result?

### Task 2: German traffic signs

Repeat the experiment with the German traffic signs (`gtsrb`) dataset, which has 5535 training images across 43 types of traffic sign.

<img src="https://raw.githubusercontent.com/csc-training/intro-to-dl/e2bdda765b52e1028cd856f49d9b07ef02f7ff8d/day2/imgs/gtsrb-montage.png" alt="Sample traffic-sign images" width="500">

<img src="https://raw.githubusercontent.com/csc-training/intro-to-dl/e2bdda765b52e1028cd856f49d9b07ef02f7ff8d/day2/imgs/traffic-signs.png" alt="The 43 traffic-sign classes" width="500">

The scripts follow the same naming as Task 1. Just replace `dvc` with `gtsrb`:

| Approach | Script |
| --- | --- |
| CNN trained from scratch | [`pytorch_gtsrb_cnn_simple.py`](pytorch_gtsrb_cnn_simple.py) |
| Pre-trained CNN (VGG16) + fine-tuning | [`pytorch_gtsrb_cnn_pretrained.py`](pytorch_gtsrb_cnn_pretrained.py) |

Then compare:

- Which model gives the best result for traffic signs?
- How do these results compare with the dogs-vs-cats results?

---

## Credits

The notebook, training scripts, example solutions, and dataset images in this
demo are adapted from CSC's
[Practical Deep Learning course materials](https://github.com/csc-training/intro-to-dl)
([csc-training/intro-to-dl](https://github.com/csc-training/intro-to-dl)). Many
thanks to the original authors.
