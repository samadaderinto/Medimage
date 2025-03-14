# Use an official Node.js image as the base
FROM node:18

# Set the working directory inside the container
WORKDIR /app

# Copy package files separately to leverage Docker cache
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install -f

# Copy the rest of the application files
COPY . .

# Build the Next.js app (for production)
RUN npm run build

# Expose the port Next.js runs on
EXPOSE 3000

# Start the Next.js application
CMD ["npm", "run", "start"]
