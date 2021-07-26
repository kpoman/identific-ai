echo "1-1.4.3" >/sys/bus/usb/drivers/usb/unbind
sleep 1
echo "1-1.4.3" >/sys/bus/usb/drivers/usb/bind
