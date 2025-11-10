# Use Node.js LTS version
FROM node:18.17-slim

# Set working directory
WORKDIR /app

# Install dependencies first (caching)
COPY package*.json ./
RUN npm install

# Copy the rest of the application, excluding .env files
COPY . .

# Set up environment variables
ARG NEXT_PUBLIC_FIREBASE_API_KEY
RUN echo "NEXT_PUBLIC_FIREBASE_API_KEY=$NEXT_PUBLIC_FIREBASE_API_KEY" > .env.local

# Set production environment
ENV NEXT_TELEMETRY_DISABLED 1
ENV NODE_ENV production

# Install platform-specific SWC binary
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      npm install @next/swc-linux-arm64-gnu @next/swc-linux-arm64-musl; \
    else \
      npm install @next/swc-linux-x64-gnu @next/swc-linux-x64-musl; \
    fi

# Build application
RUN npm run build

# Expose port 3000
EXPOSE 3000

# Start the Next.js app
CMD ["npm", "start"]