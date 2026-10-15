QwatLab - EGAD GPS Analyzer V1.0.0
Please read the license and usage requirements before using the software.
Dasturdan foydalanishdan avval litsenziya va dasturdan foydalanish talablarini oqib chiqing.

🇺🇿 Oʻzbek tilidagi tavsif
QwatLab - EGAD GPS Analyzer V1.0.0 — xom GPS (GPX, KML, CSV, XLSX) ma'lumotlarini o'qish, yuqori aniqlikdagi 3D kinematikani hisoblash, topografik relyef, ob-havo (ISA) va elektromobil (EV) harakat dinamikasini raqamli egizak (Digital Twin) formatida modellashtirish uchun mo'ljallangan Python dasturiy platformasi.
Dasturiy platforma tavsifi
Ushbu platforma GPS datchiklaridan olingan trayektoriyalarni "Dual-Pipeline" (Ikki oqimli: Xom va Filtrlangan) rejimida tahlil qilib, 9D Kalman filtrlari orqali xatolarni tozalaydi. Dastur Streamlit freymvorkiga asoslangan bo'lib, barcha matematik dvigatellar interfeysdan to'liq ajratilgan holda, sof "Qora quti" mantiqida ishlaydi. U foydalanuvchiga ob-havo va transport parametrlarini kiritish, dinamik kuchlarni hisoblash va natijalarni kompleks eksport qilish imkonini beradi.
Asosiy hisoblash va tahlil yo‘nalishlari
* Ma'lumotlarni o'qish va validatsiya: GPX, KML, CSV va XLSX fayllarini xavfsiz o'qish hamda Z-o'qi (balandlik) bo'yicha "Fail-Fast" validatsiyasi orqali anomaliyalarni aniqlash.
* 3D Kinematika: WGS84 tizimidan ECEF va ENU koordinatalariga o'tkazish, 9D RTS Kalman filtri, ZUPT (Zero Velocity Update) va Savitzky-Golay filtrlari yordamida tezlik, tezlanish, siltanish (jerk) va burilish radiusini hisoblash.
* Relyef va Atrof-muhit: Dinamik qiyalik darchasi orqali topografik qiyalikni topish, Xalqaro Standart Atmosfera (ISA) modeli yordamida bosim, harorat va havo zichligini aniqlash, shuningdek, shamol vektorlarini (qarshi va yon shamol) proyeksiyalash.
* Mexanik Dinamika: Aerodinamik qarshilik (F_{aero}), dumalash qarshiligi (F_{roll}), qiyalik kuchi (F_{grad}), inersiya (F_{acc}) va burilish (F_{corner}) qarshiliklarini hisoblash.
* Elektr uzatish va tortishuv: Ekvivalent massa, dinamik radius, og'irlik ko'chishi (h_{cg}), regenerativ tormozlanish (Brake Blending) va dvigatelning moment hamda quvvat ko'rsatkichlarini hisoblash.
* Vizualizatsiya va Eksport: Interaktiv xaritalarda ko'rsatish, Xom va Filtrlangan ma'lumotlarni qiyosiy grafiklarga solish va barcha natijalarni ZIP, CSV yoki XLSX formatida arxivlash.
Dasturiy arxitekturaning rivojlanishi va modullashtirish rejasi
1-bosqich. Modullarga bo`lish. GUI ni to`liq yadrodan ajratish:
QwatLab - EGAD GPS Analyzer dasturi "Separation of Concerns" tamoyili asosida 7 ta qat'iy izolyatsiya qilingan modul asosida ishlab chiqilgan. Hisoblash motorlari vizual interfeysdan (GUI) butunlay ajratilgan. Loyihadagi asosiy modullar quyidagilardan iborat:
1. qwt_egad_core.py – Tizimning yagona haqiqat markazi va Pydantic ma'lumotlar modellari (SystemState, KinematicsSettings va boshqalar).
2. qwt_egad_data_loader.py – Fayllarni o'qish, vaqt zanjirini tekshirish va validatsiya moduli.
3. qwt_egad_kinematics.py – Kinematika va 9D RTS Kalman filtri dvigateli.
4. qwt_egad_relief.py – Relyef qiyaligi, ob-havo va ISA model dvigateli.
5. qwt_egad_dynamics.py – Elektromobilning mexanik, transmissiya va elektr dinamikasi dvigateli.
6. qwt_egad_export.py – Eksport qilish va qiyosiy grafiklar chizish markazi.
7. qwt_egad_gui.py – Streamlit asosidagi 7 menyuli interfeys va boshqaruv dispetcheri.
2-bosqich. "Dual-Pipeline" (Ikki oqimli) hisoblash usulini ishlab chiqish.
Kiritilgan ma'lumotlar asosida tizimni xom (Raw) va filtrlangan (Filtered) ma'lumotlar oqimida parallel ishlashiga moslashtirish. Bu olimlarga GPS xatoliklari, ob-havo va topografik relyefning avtomobil energiya sarfiga ta'sirini to'g'ridan-to'g'ri qiyosiy tahlil qilish imkonini beradi.   
3-bosqich. Kengaytirish va Modulli Integratsiya:QwatLab - EGAD tizimini quyidagi asosiy motorlarga (engine) to'liq kengaytirish va izolyatsiya qilish:
Data Loader moduli: GPX, KML, CSV va XLSX fayllarini "Fail-Fast" tamoyilida o'qish va Z-o'qi validatsiyasini mustaqil bajarish.   
Kinematika moduli: WGS84 dan ECEF/ENU ga proyeksiyalash, 9D RTS Kalman filtri va ZUPT algoritmlari orqali 3D kinematikani kengaytirish.   
Relyef va Atrof-muhit moduli: Dinamik qiyalik darchasi va Xalqaro Standart Atmosfera (ISA) modellarini joriy etish.   
Dinamika moduli: Og'irlik ko'chishi (h_{cg}), ekvivalent massa (m_{eff}), barcha qarshilik kuchlari va elektr motor quvvatini hisoblash mexanikasini kengaytirish.   
Eksport va Vizualizatsiya moduli: Jadvallar, interaktiv xaritalar va grafiklar generatsiyasini hamda ZIP arxivlash jarayonini yakunlash.   
GUI Dispetcher moduli: Barcha 1-6 modullarni yagona "Single Source of Truth" (qwt_egad_core.py) orqali boshqaruvchi 7 ta menyuli interfeysni shakllantirish. 
V1.0.0 versiyasining ilmiy o‘rni
Ushbu platforma ilmiy tadqiqotlar uchun ochiq kodli muhiti sifatida xizmat qiladi. "Dual-Pipeline" (Xom va Filtrlangan) oqimli hisoblash uslubi orqali olimlar ob-havo, topografik relyef va GPS xatoliklarining elektromobil energiya sarfiga qanday ta'sir qilishini bevosita taqqoslash imkoniyatiga ega bo'ladilar.

🇬🇧 English Description
Software Platform Overview
QwatLab - EGAD GPS Analyzer V1.0.0 is a Python-based software platform designed to read raw GPS data (GPX, KML, CSV, XLSX) and model high-precision 3D kinematics, topographic relief, environmental conditions (ISA), and electric vehicle (EV) dynamics in a Digital Twin format. Operating on a Dual-Pipeline architecture (Raw and Filtered data), the system utilizes 9D Kalman filters to process trajectory data. Built on the Streamlit framework, its mathematical and physics engines function as independent "Black Boxes" fully isolated from the user interface.
Core Computational and Analytical Features
* Data Parsing and Validation: Secure extraction of GPX/KML/CSV/XLSX files with a rigorous Fail-Fast Z-axis (elevation) validation protocol to prevent critical calculation anomalies.
* 3D Kinematics Engine: WGS84 to ECEF/ENU projections, 9D RTS Kalman filtering, Zero Velocity Update (ZUPT), and Savitzky-Golay filtering to calculate velocity, acceleration, jerk, and turning radius.
* Relief & Environment: Calculating topographical grade using dynamic spatial steps, applying the International Standard Atmosphere (ISA) model for air density, pressure, and temperature, and projecting head/cross wind vectors.
* Mechanical Dynamics: Computing aerodynamic drag (F_{aero}), rolling resistance (F_{roll}), gradient force (F_{grad}), and inertial/cornering forces.
* Powertrain and Traction: Determining equivalent mass, dynamic tire radius, weight transfer (h_{cg}), regenerative braking (Brake Blending), and motor torque/power limitations.
* Visualization and Export: Presenting interactive 3D trajectories, comparing raw vs filtered parameters, and exporting comprehensive datasets to ZIP, CSV, or XLSX formats.
Software Architecture Evolution and Modularization Plan
The system is built upon 7 isolated modules following strict "Separation of Concerns" guidelines. The core computational engines are completely decoupled from the GUI layer. The project's core modules include:
Stage 1. Division into modules. Separate the full GUI from the kernel:
1. qwt_egad_core.py – Single Source of Truth, handling Pydantic validation models and shared states.
2. qwt_egad_data_loader.py – Data parsing, gap tolerance checking, and initial validation.
3. qwt_egad_kinematics.py – Kinematic calculations and 9D Kalman filter engine.
4. qwt_egad_relief.py – Topography, gradient, and ISA environment modeling.
5. qwt_egad_dynamics.py – EV mechanical and electrical powertrain dynamics.
6. qwt_egad_export.py – Result aggregation, data exporting, and static/interactive plotting.
7. qwt_egad_gui.py – Streamlit-based GUI dispatcher managing the 7-menu interface.
Stage 2. Development of the "Dual-Pipeline" computation method.
Adapting the system to process data continuously in parallel through Raw and Filtered pipelines. This enables researchers to directly compare the impact of GPS anomalies, weather, and topographic relief on the energy consumption of electric vehicles.   
Stage 3. Extension and Modular Integration:Fully expanding and isolating the QwatLab - EGAD system into the following core engines:
Data Loader Module: Implementing "Fail-Fast" extraction and Z-axis validation for GPX, KML, CSV, and XLSX formats.   
Kinematics Module: Extending 3D kinematics via WGS84 to ECEF/ENU projections, 9D RTS Kalman filtering, and ZUPT algorithms.   
Relief & Environment Module: Integrating dynamic gradient windows and the International Standard Atmosphere (ISA) models.   
Dynamics Module: Expanding mechanics to compute weight transfer (h_{cg}), equivalent mass (m_{eff}), combined resistance forces, and electric motor power limits.   
Export & Visualization Module: Finalizing the generation of tables, interactive maps, static plots, and ZIP archiving processes.   
GUI Dispatcher Module: Establishing a 7-menu interface that manages all modules through a "Single Source of Truth" (qwt_egad_core.py) core.
Scientific Role of Version 1.0.0
This platform acts as an open-source ecosystem tailored for scientific research. Through its dual-pipeline (Raw vs Filtered) computation architecture, it enables researchers to directly compare and evaluate the impact of wind, topography, and GPS signal anomalies on the dynamics and energy consumption of electric vehicles.
