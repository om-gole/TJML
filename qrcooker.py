import qrcode
import json

def create_qr_code(data, file_name):
    # Create qr code instance
    qr = qrcode.QRCode(
        version = 1,
        error_correction = qrcode.constants.ERROR_CORRECT_H,
        box_size = 10,
        border = 4,
    )

    # Convert the data to a JSON string
    json_data = json.dumps(data)

    # Add data
    qr.add_data(json_data)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    img = qr.make_image()

    # Save it
    img.save(file_name)
    print(f"QR code for {json_data} has been saved as {file_name}")

# Create a dictionary with the recipient and amount
transaction_data = {"type": "send", "recipient": "2024ogole", "amount": "5"}

create_qr_code(transaction_data, "testqr.png")
