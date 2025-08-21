FROM node:20-slim
WORKDIR /app

# Use package.json only (no lock) to avoid the ci/lockfile mismatch error
COPY package.json ./
RUN npm install --no-audit --no-fund

# Copy the rest of the app and start
COPY . .
EXPOSE 3000
CMD ["npm","run","dev"]
