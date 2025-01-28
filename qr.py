import qrcode
from PIL import Image

def url_to_ascii_qr(url, box_size=2, scale=1):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")

    img.save("qr.png")
    img = img.convert("L")
    
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale * 0.5) 
    img = img.resize((new_width, new_height))
    
    ascii_str = ""
    
    for y in range(new_height):
        for x in range(new_width):
            pixel = img.getpixel((x, y))
            ascii_str += "  " if pixel > 128 else "██"
        ascii_str += "\n"
    
    return ascii_str

if __name__ == "__main__":
    url = input("Masukkan URL: ")
    ascii_qr = url_to_ascii_qr(url, box_size=4, scale=0.5)
    print(ascii_qr)