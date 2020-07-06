# machine-learning
Deri kanseri dünya çapında en yaygın kanserlerdendir ve melanoma bu kanser türünün en ölümcül
formdur. Ödev kapsamında size iki adet “.csv” uzantılı dosya verilmiştir. İlk dosya olan meta-data
dosyasında image, age_approx, anatom_site_general, lesion_id, gender gibi verilerin yer aldığı
kolonlar bulunmaktadır. Diğer dosya olan ground_truth dosyasında ise her bir görüntüye ait sınıf
kategorisi bulunmaktadır. Oradaki kolonlar ise image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
oluşmaktadır. Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis
, Dermatofibroma, Vascular lesion, Squamous cell carcinoma, None of the others gibi farklı sınıftan
oluşmaktadır.
Örneğin metadata dosyasında ilk satırda yer alan veriye ve ground_truth dosyasının ilk satırı
birleşince aşağıdaki gibi olmaktadır.
ISIC_0000000,55,anterior torso,,female - ISIC_0000000,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Görüldüğü üzere ISIC_0000000 nolu örnek 55 yaşında bayan bir hastaya aittir. Lezyonun bulunduğu
bölge ise anterior torso olarak belirtilmiştir. Dokuz kategori arasından örnek ikili kodlamaya göre NV
olarak kodlanmıştır.
1- Analiz için size gerekli olan özellikler image, age_approx, anatom_site_general ve gender olacaktır.
Meta-data veri seti içinde kalan harici özellikler veriden silinecektir.
2- Örnekler arasında belirtilen üç özellikten herhangi biri boş ise söz konusu örnek nihai veri setine
alınmayacaktır. Örneğin ISIC_0000000, ,anterior torso,,female şeklinde yaş kolonu boş ise bu örnek
nihai veri setine alınmayacaktır. Bu temizlikten sonra boş veriler atılacaktır.
3- Metadata dosyasındaki veriler temizlendikten sonra gorund_truth dosyası ile ilişkilendirmesi
yapılacak her bir örneğin karşısına ait olduğu sınıf yazılacaktır. Son durumda elimizde
ISIC_0000000,55,anterior torso,female, NV gibi bir örnek yer alacaktır.
4- Veriyi sınıflandırma için bazı hazırlıklar yapılması gerekmektedir. Bu hazırlıklar kategorik verilerin
nümerik hale getirilmesi, sayısal verilerin ise ölçeklenmesidir.
5- Elde edeceğiniz son veriyi eğitim ve test olmak üzere %80, %20 olacak şekilde bölün ve
öğrendiğimiz yöntemleri eğitim verisi ile eğitin. Test verisi ile testlerini yapıp accuracy, sensitivity,
specifisity, presicion, recall ve f1 score ölçütlerini  Ayrıca her bir yöntemin elde ettiği
karmaşıklık matrislerini gösterilmiştir.
