import os
from dotenv import load_dotenv
import mysql.connector

# Charger le .env
load_dotenv()

print("=== TEST CONNEXION MYSQL ===")
print(f"HOST: '{os.getenv('MYSQL_HOST')}'")
print(f"PORT: '{os.getenv('MYSQL_PORT')}'") 
print(f"USER: '{os.getenv('MYSQL_USER')}'")
print(f"PASSWORD: '{'*' * len(os.getenv('MYSQL_PASSWORD', ''))}' (length: {len(os.getenv('MYSQL_PASSWORD', ''))})")
print(f"DATABASE: '{os.getenv('MYSQL_DATABASE')}'")

try:
    connection = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        port=int(os.getenv('MYSQL_PORT', '3306')),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        database=os.getenv('MYSQL_DATABASE', 'bddelections')
    )
    print("\n✅ CONNEXION RÉUSSIE !")
    
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM resultatslegi")
    count = cursor.fetchone()[0]
    print(f"📊 Nombre d'enregistrements : {count}")
    
    connection.close()
    
except Exception as e:
    print(f"\n❌ ERREUR : {e}")