model.eval()
 test_img1 = transform(Image.open("test_examples/dog.jpg").conver
t("RGB")).unsqueeze(
 0
 )
 print("Example 1 CORRECT: Dog on a beach by the ocean")
 print(
 "Example 1 OUTPUT: "
 + " ".join(model.caption_image(test_img1.to(device), dataset
.vocab))
 )
 test_img2 = transform(
 Image.open("test_examples/child.jpg").convert("RGB")
 ).unsqueeze(0)
 print("Example 2 CORRECT: Child holding red frisbee outdoors")
 print(
 "Example 2 OUTPUT: "
 + " ".join(model.caption_image(test_img2.to(device), dataset
.vocab))
 )
 test_img3 = transform(Image.open("test_examples/bus.png").conver
t("RGB")).unsqueeze(
 0
 )
 print("Example 3 CORRECT: Bus driving by parked cars")
 print(
 "Example 3 OUTPUT: "
 + " ".join(model.caption_image(test_img3.to(device), dataset
.vocab))
 )
