FROM python:3.11

# Create and use a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Set Hugging Face cache directory to persist model downloads
ENV HF_HOME="/home/user/.cache/huggingface"
ENV SENTENCE_TRANSFORMERS_HOME="/home/user/.cache/huggingface/sentence-transformers"

# Create cache directories and ensure permissions
RUN mkdir -p $SENTENCE_TRANSFORMERS_HOME && \
    chown -R user:user /home/user/.cache/huggingface

# Download and persist the model properly (USE snapshot_download)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/model_cache')"

# Ensure ownership and permissions remain intact
RUN chown -R user:user /app/model_cache

# Expose application port
EXPOSE 7860

# Run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
