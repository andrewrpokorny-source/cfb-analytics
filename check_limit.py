import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")

def main():
    print("--- 🕵️ CHECKING API STATUS ---")
    
    if not API_KEY:
        print("❌ Error: No API Key found in .env")
        return

    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        # The /info endpoint returns your specific usage stats
        res = requests.get("https://api.collegefootballdata.com/info", headers=headers)
        
        if res.status_code == 200:
            data = res.json()
            print("\n✅ API Key is Valid!")
            print(f"👤 Name:     {data.get('name')}")
            print(f"📧 Email:    {data.get('email')}")
            print(f"📊 Tier:     {data.get('tier')}")
            
            # Print Usage Stats
            # Note: The structure might vary, but standard keys usually appear here
            print("-" * 30)
            print("FULL RESPONSE (Look for 'limit' or 'remaining'):")
            print(data)
            
        elif res.status_code == 429:
             print("\n❌ 429 Error: You are currently blocked.")
             print("   If this happens on a single check, you are likely Rate Limited (speed).")
             print("   If it happens continuously, you might be out of Monthly Quota.")
             
        else:
            print(f"\n⚠️ Error {res.status_code}: {res.text}")

    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    main()