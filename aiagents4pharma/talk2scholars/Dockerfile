FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies required for compiling packages
RUN apt-get update && apt-get install -y \
  g++ \
  build-essential \
  cmake \
  libopenblas-dev \
  libomp-dev \
  ninja-build \
  && rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY aiagents4pharma/talk2scholars /app/aiagents4pharma/talk2scholars
COPY docs /app/docs
COPY app /app/app
COPY requirements.txt /app

# Install Python dependencies and FAISS via Conda
RUN conda update -n base -c defaults conda -y \
    && conda install -n base python=3.12 -y -c conda-forge \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && conda install -n base -c pytorch -c conda-forge faiss-cpu -y \
    && conda clean -afy

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app/frontend/streamlit_app_talk2scholars.py", "--server.port=8501", "--server.address=0.0.0.0"]
