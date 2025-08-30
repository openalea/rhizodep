# Multistage build

# Stage 1 : Full build with git, pip, source files

# 25/07/28 micromamba version
FROM mambaorg/micromamba:2.3.1 AS builder

# micromamba needs prefix to behave as an initiated conda shell
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/envs/rootcynaps/bin:$PATH

# Installing git needed to retreive the model
USER root
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone only latest snapshot or model packages and their dependancies
RUN mkdir /home/package && cd /home/package && \
    git clone --depth 1 https://github.com/openalea/Root-CyNAPS.git && \
    git clone --depth 1 https://github.com/openalea/metafspm.git && \
    git clone --depth 1 https://github.com/openalea/fspm-utility.git

# Install Conda env + dependencies + pip packages + cleanup
RUN micromamba create -y -n rootcynaps -f /home/package/Root-CyNAPS/conda/environment.yaml && \
    micromamba install -y -n rootcynaps -c conda-forge jupyterlab && \
    micromamba run -n rootcynaps pip install --no-cache-dir /home/package/Root-CyNAPS && \
    micromamba run -n rootcynaps pip install --no-cache-dir /home/package/metafspm && \
    micromamba run -n rootcynaps pip install --no-cache-dir /home/package/fspm-utility && \
    # Clean up extra space
    micromamba run -n rootcynaps pip cache purge && \
    micromamba clean --all --yes


    
# Stage 2: Runtime with strict minimum to try reducing image size
FROM mambaorg/micromamba:2.3.1 AS runtime

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/envs/rootcynaps/bin:$PATH

# Copy just the environment from builder stage
COPY --from=builder /opt/conda/envs/rootcynaps /opt/conda/envs/rootcynaps

# GUI libraries (needed at runtime)
USER root
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy notebook and input files only
RUN mkdir /home/notebook
COPY --from=builder /home/package/Root-CyNAPS/doc/notebooks /home/notebook

EXPOSE 8888
WORKDIR /home/notebook

CMD ["micromamba", "run", "-n", "rootcynaps", "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", \
     "--notebook-dir=/home/notebook", \
     "/home/notebook/example_notebook_24h_static.ipynb"]