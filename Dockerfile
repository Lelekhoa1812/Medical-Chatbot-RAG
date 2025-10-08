FROM python:3.11

# Create and use a non-root user (optional)
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
ENV MEDGEMMA_HOME="/home/user/.cache/huggingface/sentence-transformers"

# Create cache directories and ensure permissions
RUN mkdir -p /app/model_cache /home/user/.cache/huggingface/sentence-transformers && \
    chown -R user:user /app/model_cache /home/user/.cache/huggingface

# Pre-load model in a separate script
RUN python /app/models/download_model.py && python /app/models/warmup.py

# Ensure ownership and permissions remain intact
RUN chown -R user:user /app/model_cache

# Expose port
EXPOSE 7860

# Run the application using main.py as entry point
CMD ["python", "main.py"]
