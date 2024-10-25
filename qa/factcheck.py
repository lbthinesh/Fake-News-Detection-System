import requests
from datetime import date
from django.conf import settings


API_KEY = settings.FACT_CHECK_API_KEY

QUERY = "a"
URL = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?key={API_KEY}&query={QUERY}&maxAgeDays=2&languageCode=en"

def fact_checker():
    response = requests.get(URL)
    data = response.json()
    for claim in data.get('claims', []):
        if claim.get('claimDate',[])[:10]:
            claim_=claim.get('text')
            review = claim.get('claimReview',[None])[0]
            rating_=review['textualRating']
            url_=review['url']
            claimdate_=claim.get('claimDate',[])[:10]
            
            return [claim_,rating_,claimdate_,url_]
