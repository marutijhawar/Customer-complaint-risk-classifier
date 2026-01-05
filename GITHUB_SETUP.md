# GitHub Setup Guide

Follow these steps to upload your project to GitHub.

## Prerequisites

- GitHub account ([create one](https://github.com/join) if needed)
- Git installed on your computer

## Step-by-Step Instructions

### 1. Create a New Repository on GitHub

1. Go to [GitHub](https://github.com)
2. Click the **+** icon in the top right → **New repository**
3. Fill in the details:
   - **Repository name**: `customer-complaint-risk-classifier`
   - **Description**: `NLP system for classifying customer complaint risk levels`
   - **Visibility**: Choose Public or Private
   - **DO NOT** check "Initialize with README" (we already have one)
4. Click **Create repository**

### 2. Initialize Git in Your Project

Open Terminal and navigate to the project folder:

```bash
cd "/Users/marutijhawar/Desktop/desktop/doll websites /NLP project /customer-complaint-risk-classifier-github"
```

Initialize Git:

```bash
git init
```

### 3. Add All Files

```bash
git add .
```

### 4. Create First Commit

```bash
git commit -m "Initial commit: Customer Complaint Risk Classifier"
```

### 5. Link to GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/customer-complaint-risk-classifier.git
```

### 6. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

### 7. Verify Upload

1. Go to your GitHub repository URL
2. You should see all files uploaded
3. The README.md will be displayed on the main page

## Optional: Add Repository Topics

On your GitHub repository page:
1. Click the ⚙️ icon next to "About"
2. Add topics: `nlp`, `machine-learning`, `customer-service`, `risk-classification`, `python`, `xgboost`, `scikit-learn`
3. Save changes

## Optional: Enable GitHub Pages (for Documentation)

1. Go to repository **Settings**
2. Scroll to **Pages** section
3. Under "Source", select `main` branch
4. Click Save
5. Your documentation will be available at: `https://YOUR_USERNAME.github.io/customer-complaint-risk-classifier/`

## Troubleshooting

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/customer-complaint-risk-classifier.git
```

### Error: "failed to push"
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Authentication Issues

If using HTTPS and getting authentication errors:
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` permissions
3. Use the token as your password when pushing

Or switch to SSH:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/customer-complaint-risk-classifier.git
```

## Next Steps

- Add a description and topics to your repository
- Create GitHub Issues for future features
- Share your repository link!

---

**Your project is now on GitHub! 🎉**
