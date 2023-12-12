###############################################################################
# EXTRA CODE

###############################################################################
if 0: # ADD MASK TO RGBA
    rendered = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/in.png")
    rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGBA)
    mask = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/mask.png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    rendered[:,:,3] = mask[:,:,0]
    rendered = cv2.cvtColor(rendered, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/in2.png",rendered)

###############################################################################
# CLASS version

# class _Loss:
#     def __init__(name):
#         self.name = name
#     def loss(target, input, **kwargs):
#         pass

# class MSE(_Loss):
#     def __init__():
#         super().__init__(__name__)
#     def loss(target, input, h):
#         return target


# class MSE(_Loss):
#     def __init__():
#         super().__init__(__name__)
#     def loss(target, input, h):
#         return target

#losses = [
    #MSE(),
    #L1()
#]
