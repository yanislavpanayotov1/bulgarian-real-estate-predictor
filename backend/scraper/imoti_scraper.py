import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from typing import List, Dict, Optional
import re
from geopy.geocoders import Nominatim
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImotiScraper:
    """
    Web scraper for imoti.bg - Bulgarian real estate portal
    Collects property data including prices, locations, and features
    """
    
    def __init__(self):
        self.base_url = "https://www.imoti.bg"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.geolocator = Nominatim(user_agent="real_estate_scraper")
        self.properties = []
        
    def get_property_listings(self, city: str = "sofia", property_type: str = "apartments", 
                            pages: int = 5) -> List[str]:
        """Get property listing URLs from search results"""
        listings = []
        
        for page in range(1, pages + 1):
            url = f"{self.base_url}/pcgi/imoti.cgi?act=3&slink=7s11r3&f1={city}&f4={property_type}&f47=1&f44={page}"
            
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html5lib')
                
                # Find property links
                property_links = soup.find_all('a', href=re.compile(r'/pcgi/imoti\.cgi\?act=5'))
                
                for link in property_links:
                    href = link.get('href')
                    if href:
                        full_url = self.base_url + href if href.startswith('/') else href
                        listings.append(full_url)
                
                logger.info(f"Found {len(property_links)} properties on page {page}")
                
                # Random delay to avoid being blocked
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {str(e)}")
                continue
                
        return list(set(listings))  # Remove duplicates
    
    def extract_property_details(self, url: str) -> Optional[Dict]:
        """Extract detailed information from a property listing"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html5lib')
            
            property_data = {
                'url': url,
                'price': self._extract_price(soup),
                'location': self._extract_location(soup),
                'size': self._extract_size(soup),
                'rooms': self._extract_rooms(soup),
                'floor': self._extract_floor(soup),
                'year_built': self._extract_year_built(soup),
                'features': self._extract_features(soup),
                'description': self._extract_description(soup),
                'coordinates': None,  # Will be geocoded later
                'neighborhood': None,
                'city': None
            }
            
            # Parse location details
            if property_data['location']:
                location_parts = property_data['location'].split(',')
                if len(location_parts) >= 2:
                    property_data['city'] = location_parts[0].strip()
                    property_data['neighborhood'] = location_parts[1].strip()
            
            return property_data
            
        except Exception as e:
            logger.error(f"Error extracting property details from {url}: {str(e)}")
            return None
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract price from property listing"""
        price_patterns = [
            r'(\d+(?:\s?\d+)*)\s*лв',
            r'(\d+(?:\s?\d+)*)\s*EUR',
            r'(\d+(?:\s?\d+)*)\s*€'
        ]
        
        text = soup.get_text()
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(' ', '').replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    continue
        return None
    
    def _extract_location(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract location information"""
        location_selectors = [
            'td:contains("Населено място")',
            'td:contains("Локация")',
            '.location',
            '.address'
        ]
        
        for selector in location_selectors:
            elements = soup.select(selector)
            for element in elements:
                next_sibling = element.find_next_sibling()
                if next_sibling and next_sibling.get_text().strip():
                    return next_sibling.get_text().strip()
        
        return None
    
    def _extract_size(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract property size in square meters"""
        size_pattern = r'(\d+(?:\.\d+)?)\s*м²?'
        text = soup.get_text()
        match = re.search(size_pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_rooms(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract number of rooms"""
        rooms_patterns = [
            r'(\d+)\s*(?:стаи|стая|rooms?)',
            r'(\d+)\s*-?\s*(?:стаен|стайен)'
        ]
        
        text = soup.get_text()
        for pattern in rooms_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _extract_floor(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract floor number"""
        floor_pattern = r'(\d+)\s*(?:етаж|floor)'
        text = soup.get_text()
        match = re.search(floor_pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_year_built(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract year built"""
        year_pattern = r'(?:построен|built).*?(\d{4})'
        text = soup.get_text()
        match = re.search(year_pattern, text, re.IGNORECASE)
        if match:
            try:
                year = int(match.group(1))
                if 1800 <= year <= 2024:
                    return year
            except ValueError:
                pass
        return None
    
    def _extract_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract property features and amenities"""
        features = []
        feature_keywords = [
            'паркинг', 'parking', 'балкон', 'balcony', 'тераса', 'terrace',
            'асансьор', 'elevator', 'климатик', 'air conditioning', 'централно',
            'heating', 'furnished', 'обзаведен', 'гараж', 'garage'
        ]
        
        text = soup.get_text().lower()
        for keyword in feature_keywords:
            if keyword in text:
                features.append(keyword)
        
        return features
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property description"""
        desc_selectors = [
            '.description',
            '.details',
            'td:contains("Описание")'
        ]
        
        for selector in desc_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if len(text) > 50:  # Ensure it's a meaningful description
                    return text[:500]  # Limit length
        
        return None
    
    def geocode_properties(self):
        """Add coordinates to properties using geocoding"""
        logger.info("Starting geocoding of properties...")
        
        for i, prop in enumerate(self.properties):
            if prop.get('location'):
                try:
                    location = f"{prop['location']}, Bulgaria"
                    geocoded = self.geolocator.geocode(location, timeout=10)
                    
                    if geocoded:
                        prop['coordinates'] = {
                            'lat': geocoded.latitude,
                            'lng': geocoded.longitude
                        }
                        logger.info(f"Geocoded property {i+1}/{len(self.properties)}")
                    else:
                        logger.warning(f"Could not geocode: {location}")
                    
                    # Rate limiting for geocoding service
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Geocoding error for {prop['location']}: {str(e)}")
                    continue
    
    def scrape_properties(self, cities: List[str] = ["sofia", "plovdiv", "varna"], 
                         pages_per_city: int = 10) -> pd.DataFrame:
        """Main scraping method"""
        logger.info("Starting property scraping...")
        
        all_listings = []
        
        # Collect listing URLs
        for city in cities:
            logger.info(f"Scraping listings for {city}...")
            listings = self.get_property_listings(city, pages=pages_per_city)
            all_listings.extend(listings)
            time.sleep(2)
        
        logger.info(f"Found {len(all_listings)} total property listings")
        
        # Extract details from each listing
        for i, url in enumerate(all_listings[:500]):  # Limit to 500 properties for demo
            logger.info(f"Processing property {i+1}/{min(len(all_listings), 500)}")
            
            property_data = self.extract_property_details(url)
            if property_data and property_data['price']:  # Only keep properties with prices
                self.properties.append(property_data)
            
            # Rate limiting
            time.sleep(random.uniform(2, 4))
        
        # Geocode properties
        self.geocode_properties()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.properties)
        
        # Clean and validate data
        df = self._clean_data(df)
        
        # Save to file
        output_dir = Path("../data")
        output_dir.mkdir(exist_ok=True)
        
        df.to_csv(output_dir / "raw_properties.csv", index=False)
        logger.info(f"Saved {len(df)} properties to raw_properties.csv")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate scraped data"""
        logger.info("Cleaning scraped data...")
        
        # Remove rows without essential data
        df = df.dropna(subset=['price', 'location'])
        
        # Convert price to numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Remove unrealistic prices (less than 1000 or more than 10M EUR)
        df = df[(df['price'] >= 1000) & (df['price'] <= 10000000)]
        
        # Clean size data
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df = df[df['size'] > 10]  # Remove unrealistic sizes
        
        # Clean rooms data
        df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
        
        # Calculate price per square meter
        df['price_per_sqm'] = df['price'] / df['size']
        
        # Remove duplicates based on URL
        df = df.drop_duplicates(subset=['url'])
        
        logger.info(f"Cleaned data: {len(df)} properties remaining")
        
        return df

def main():
    """Main function to run the scraper"""
    scraper = ImotiScraper()
    
    # Scrape properties from major Bulgarian cities
    cities = ["sofia", "plovdiv", "varna", "burgas", "stara-zagora"]
    
    try:
        df = scraper.scrape_properties(cities=cities, pages_per_city=5)
        print(f"Successfully scraped {len(df)} properties")
        print(df.head())
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(f"Average price: {df['price'].mean():.2f} BGN")
        print(f"Average size: {df['size'].mean():.2f} m²")
        print(f"Average price per m²: {df['price_per_sqm'].mean():.2f} BGN/m²")
        
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")

if __name__ == "__main__":
    main() 