# 🚀 **FREE DEPLOYMENT GUIDE**

Deploy your **Bulgarian Real Estate Price Predictor** to the world for **FREE**!

---

## 🎯 **Deployment Strategy**

- **Frontend** → **Vercel** (Perfect for Next.js, free tier)
- **Backend** → **Railway** (Great for ML apps, free $5/month credit)

**Total Cost: $0/month** (within free limits)

---

## 📋 **Prerequisites**

1. **GitHub Account** (for connecting to deployment platforms)
2. **Email Account** (for platform signups)
3. **Your working app** (which you have! ✅)

---

## 🚀 **STEP 1: Deploy Backend to Railway**

### **1.1 Prepare Backend**
```bash
./deploy_backend.sh
```

### **1.2 Setup Railway Account**
1. Go to **https://railway.app**
2. Click **"Start a New Project"**  
3. Sign up with **GitHub** (recommended)
4. Verify your email

### **1.3 Deploy Backend**
```bash
cd backend

# Login to Railway
railway login

# Initialize project
railway init
# Choose: "Empty Project"
# Project name: "bulgarian-real-estate-api"

# Deploy!
railway up
```

### **1.4 Get Your Backend URL**
- After deployment, Railway will show your URL
- Should look like: `https://bulgarian-real-estate-api-production.railway.app`
- **Save this URL!** You'll need it for the frontend.

### **1.5 Test Backend**
```bash
curl https://your-railway-url.railway.app/health
curl https://your-railway-url.railway.app/properties?limit=3
```

---

## 🎨 **STEP 2: Deploy Frontend to Vercel**

### **2.1 Prepare Frontend**
```bash
./deploy_frontend.sh
```

### **2.2 Setup Vercel Account**
1. Go to **https://vercel.com**
2. Click **"Sign Up"**
3. Sign up with **GitHub** (recommended)

### **2.3 Deploy Frontend**
```bash
cd frontend

# Deploy to Vercel
vercel

# Follow prompts:
# - Link to existing project? No
# - What's your project's name? bulgarian-real-estate-predictor
# - In which directory is your code located? ./
# - Want to override settings? No
```

### **2.4 Set Environment Variable**
```bash
# Set your backend URL
vercel env add NEXT_PUBLIC_API_URL
# Enter: https://your-railway-url.railway.app

# Redeploy with new environment variable
vercel --prod
```

---

## ✅ **STEP 3: Test Your Deployed App**

### **Your live URLs:**
- **Frontend**: https://your-project-name.vercel.app
- **Backend**: https://your-project-name.railway.app

### **Test checklist:**
- ✅ Frontend loads
- ✅ Properties show on map (even if clustered in Sofia)
- ✅ Price prediction works
- ✅ Market statistics display

---

## 🛠️ **Alternative Deployment Options**

### **If Railway doesn't work:**

#### **Option B: Render**
```bash
# 1. Go to render.com
# 2. Connect GitHub repository
# 3. Choose "Web Service"
# 4. Build Command: pip install -r requirements.txt
# 5. Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

#### **Option C: Heroku**
```bash
# 1. Install Heroku CLI
# 2. heroku create your-app-name
# 3. git push heroku main
```

### **If Vercel doesn't work:**
- **Netlify** - Similar to Vercel, great for React/Next.js
- **Vercel alternatives**: Surge.sh, GitHub Pages

---

## 💰 **Free Tier Limits**

### **Vercel (Frontend)**
- ✅ **Unlimited** personal projects
- ✅ **100GB** bandwidth/month  
- ✅ **Custom domains**

### **Railway (Backend)**
- ✅ **$5 credit** per month (usually enough!)
- ✅ **500GB** bandwidth
- ✅ **Up to 8GB** RAM

### **Usage estimates for your app:**
- **Backend**: ~$2-3/month (well within free credit)
- **Frontend**: Free (unless huge traffic)

---

## 🐛 **Troubleshooting**

### **Backend deployment fails:**
```bash
# Check logs
railway logs

# Common fixes:
# 1. Ensure requirements.txt is correct
# 2. Check if models directory copied
# 3. Verify Python version (should be 3.11+)
```

### **Frontend can't connect to backend:**
```bash
# Check environment variable
vercel env ls

# Should show: NEXT_PUBLIC_API_URL = your-railway-url

# If missing, add it:
vercel env add NEXT_PUBLIC_API_URL
vercel --prod  # Redeploy
```

### **CORS errors:**
- Backend already configured for CORS
- If issues persist, check if Railway URL uses HTTPS

---

## 🎉 **Post-Deployment**

### **Share your app:**
- Frontend URL is public and shareable!
- Add it to your portfolio, LinkedIn, resume

### **Monitor usage:**
- **Railway**: Check dashboard for usage
- **Vercel**: Monitor in dashboard

### **Future updates:**
```bash
# Update backend
cd backend
railway up

# Update frontend  
cd frontend
vercel --prod
```

---

## 🌟 **Your App is Now LIVE!**

**Congratulations!** 🎊 

Your **Bulgarian Real Estate Price Predictor** is now deployed and accessible worldwide!

- **ML-powered predictions** ✅
- **Interactive map** ✅  
- **Professional UI** ✅
- **Free hosting** ✅

**Perfect for your portfolio, resume, or sharing with others!** 🚀

---

## 📞 **Need Help?**

If you run into any issues during deployment, let me know:
1. **Which step failed?**
2. **Error messages you see**
3. **Platform you're using** (Railway, Vercel, etc.)

I'll help you get it deployed! 🎯 