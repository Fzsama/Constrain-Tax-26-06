"""种子询盘场景模板库 — 30 个多样询盘场景。

覆盖维度：
- 行业: 10 类，每类 3 个
- 地区: 7 大区域均匀分布
- 买家类型: 品牌商 / 分销商 / 零售商
- 复杂度: 简单RFQ / 标准询盘 / 复杂询盘

每个场景包含元数据 + 完整 user_message（模拟用户输入给 EvoAgent 的内容）。
"""

from __future__ import annotations

from typing import List

# ============================================================
# 场景数据结构
# ============================================================
# 每个场景 = {id, industry, region, country, buyer_type, complexity, user_message}


def _make_inquiry(
    sid: str,
    industry: str,
    region: str,
    country: str,
    buyer_type: str,
    complexity: str,
    message: str,
) -> dict:
    return {
        "id": sid,
        "industry": industry,
        "region": region,
        "country": country,
        "buyer_type": buyer_type,
        "complexity": complexity,
        "user_message": message.strip(),
    }


SEED_SCENARIOS: List[dict] = []

# ============================================================
# 1. LED 照明 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_001", "LED 照明", "西欧", "德国", "品牌商", "标准询盘",
    """请分析这封询盘:

公司: LichtDesign GmbH, 德国高端照明品牌商
产品: LED Strip Lights, 24V, CRI>90, 2700K-6500K tunable white, 10mm PCB, IP20
数量: 5000 meters (1000 rolls × 5m/roll)
要求: DDP 汉堡港交货, CE/ROHS/REACH 认证, OEM 中性包装, 期望 45 天交期
补充: 客户官网 www.lichtdesign.de，签名中有 Procurement Manager 头衔"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_002", "LED 照明", "中东", "阿联酋", "分销商", "简单RFQ",
    """分析这个迪拜客户的询盘:

Hi, we are Al-Noor Lighting Trading LLC, based in Dubai. We saw your LED downlights.
Please quote:
- 10W COB LED Downlight, Warm White 3000K, Cutout 90mm
- Qty: 2000 pcs for first trial
- FOB price and delivery time to Jebel Ali Port
- Do you have SASO certificate?

Best regards,
Ahmed"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_003", "LED 照明", "北美", "美国", "品牌商", "复杂询盘",
    """分析这个美国客户的详细询盘:

From: procurement@brightlight-usa.com
Company: BrightLight Innovations Inc., Orlando FL

We are developing a new line of smart LED floodlights for the US market and need a reliable OEM partner.

Technical requirements:
- 100W/200W/300W LED Floodlight, smart WiFi + Bluetooth Mesh
- CCT tunable 3000K-6500K, CRI≥80
- IP66 waterproof, IK08 impact rating
- Input: 100-277V AC, PF>0.95
- Must support Alexa/Google Home integration
- UL/ETL listed (mandatory), FCC Part 15, Energy Star

Initial order: 500 pcs each wattage as samples, then 40ft container monthly
Target FOB price: under $50 for 100W model
Payment: L/C at sight or T/T 30% advance + 70% against B/L

Please provide:
1. Your company profile and US customer references
2. Specification sheets for comparable models
3. Timeline for UL certification if not already certified
4. MOQ and lead time

公司网站: www.brightlight-usa.com"""
))

# ============================================================
# 2. 消费电子 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_004", "消费电子", "北美", "美国", "分销商", "复杂询盘",
    """分析这个美国客户的询盘:

Company: TechDist Inc., San Jose CA
Contact: John Zhang, Sourcing Director

We are a leading B2B electronics distributor in Silicon Valley, supplying to corporate clients including Google and Meta.

Products needed:
A) 12-in-1 USB-C Hub: 2×HDMI 4K@60Hz, 1×DP 1.4, 3×USB-A 3.2, 2×USB-C PD, SD/TF, RJ45 2.5G, 3.5mm Audio, PD 100W pass-through
B) 8-in-1 USB-C Hub: 1×HDMI 4K@60Hz, 2×USB-A 3.0, 1×USB-C, SD/TF, RJ45 1G, PD 100W

Quantities: 3000 units each model for first order
Target price: under $22 (8-in-1) / $35 (12-in-1) FOB Shenzhen
Certifications: UL, FCC, CE
Payment: Net 30 after delivery (D&B credit check available)

Please reply with:
1. Your existing USB-C hub models and pricing
2. OEM/ODM capability (custom housing, logo, packaging)
3. Lead time for 6000 units
4. Warranty terms

www.techdist.com"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_005", "消费电子", "东南亚", "印度", "零售商/终端用户", "标准询盘",
    """分析这封印度客户的询盘:

Dear Sir,
We are GadgetWorld India Pvt Ltd, a retail chain with 50+ stores across Mumbai and Delhi.
We are interested in your TWS Bluetooth earbuds.

Requirements:
- Bluetooth 5.3, ENC noise cancellation
- Touch control, IPX5 water resistant
- Battery life: 6hrs buds + 30hrs case
- USB-C charging, wireless charging optional
- Color: Black and White

Quantity: 5000 pairs (2500 each color)
Target price: under $8/pair FOB
Delivery: 30 days

We need samples first. Do you have BIS certification?

www.gadgetworld.in"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_006", "消费电子", "西欧", "法国", "品牌商", "标准询盘",
    """分析这封法国客户的询盘:

Bonjour,
We are Tech'Innov, a French consumer electronics brand based in Paris.
We are looking for a manufacturer of GaN USB-C chargers.

Specifications:
- 65W and 100W models, GaN technology
- 2×USB-C + 1×USB-A ports
- Compact design, foldable plug (EU plug)
- Support PD 3.0, QC 4.0+, PPS protocols
- CE, RoHS, ErP Lot 6 certified

Quantity: 10,000 units (8000 × 65W + 2000 × 100W)
We require full French retail packaging with our branding.

Pouvez-vous nous envoyer votre catalogue et meilleur prix?

www.techinnov.fr"""
))

# ============================================================
# 3. 机械设备 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_007", "机械设备", "南美", "巴西", "分销商", "复杂询盘",
    """分析这个巴西客户的询盘:

Empresa: MaqTech Industrial Ltda, São Paulo, Brasil
Contato: Ricardo Silva, Diretor de Compras

We are specialized in plastic injection molding machines distribution in Brazil
and are looking to expand our product portfolio with Chinese manufacturers.

Requirements for Injection Molding Machine:
- Clamping force: 200T, 350T, 500T
- Injection weight: 200g-2000g (depending on model)
- Servo motor energy-saving system
- PLC control with touch screen, multi-language (Portuguese required)
- Must have INMETRO certification or willingness to obtain
- CE/ISO 9001 factory certification

Expected annual volume: 30-50 units per year across all models
Initial trial order: 3 units (1 × 200T + 1 × 350T + 1 × 500T)
Incoterms: CIF Santos Port
Payment: 30% advance + 70% against shipment documents

We will visit China next month and would like to visit your factory.

www.maqtech.com.br"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_008", "机械设备", "中东", "沙特阿拉伯", "零售商/终端用户", "标准询盘",
    """分析这个沙特客户的询盘:

Dear Manufacturer,
We are Al-Jazirah Packaging Factory in Riyadh, Saudi Arabia.
We need a CNC Router for our packaging material cutting line.

Machine specification:
- Working area: 1300mm × 2500mm
- Spindle: 9KW HSD air cooling
- Control: DSP A18 or Syntec
- Application: cutting acrylic, MDF, ACM panels for signage
- Vacuum table with T-slot

Quantity: 1 unit for evaluation, potential for 10+ units for our new factory
SASO Certificate of Conformity required
CIF Dammam Port

Please send specification and quotation.

Regards,
Khalid Al-Otaibi
Procurement Engineer"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_009", "机械设备", "东南亚", "越南", "品牌商", "标准询盘",
    """分析这个越南客户的询盘:

Kính gửi Nhà cung cấp,

Công ty chúng tôi là VinaPack Co., Ltd, chuyên sản xuất bao bì carton tại Bắc Ninh, Việt Nam.
Chúng tôi cần mua máy in flexo cho dây chuyền sản xuất thùng carton.

Yêu cầu kỹ thuật:
- Máy in flexo 3 màu, khổ in tối đa 1600mm
- Tốc độ: 150-200 tờ/phút
- Có chức năng cắt rãnh và đục lỗ
- Hệ thống điều khiển PLC
- Điện áp: 380V/50Hz/3 pha (tiêu chuẩn Việt Nam)

Số lượng: 2 máy
Thời gian giao hàng: 60 ngày
Điều kiện giao hàng: CIF Hai Phong Port

Chúng tôi muốn biết giá và thời gian lắp đặt/hướng dẫn vận hành.

www.vinapack.vn"""
))

# ============================================================
# 4. 纺织服装 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_010", "纺织服装", "西欧", "意大利", "品牌商", "复杂询盘",
    """分析这个意大利客户的询盘:

Company: ModaMilano SRL, Milan, Italy
Contact: Francesca Rossi, Head of Sourcing

We are a premium fashion brand producing seasonal collections.
We need a partner for our upcoming Spring/Summer 2026 collection.

Products:
A) Women's silk blouses: 100% mulberry silk 19mm, digital print, 3 designs × 500pcs each
B) Women's linen trousers: 100% European flax linen, garment-dyed, 4 colors × 300pcs each
C) Women's cotton poplin shirts: 100% GOTS organic cotton, 2 designs × 800pcs each

Requirements:
- OEKO-TEX Standard 100 certification
- GOTS certification for organic cotton items
- BSCI or SEDEX audited factory
- European sizing with graded spec sheets
- Full package: fabric sourcing → production → packing with hang tags/barcode

Timeline: samples by August 2026, bulk delivery by December 2026
Payment: 30% deposit + 70% against B/L
Incoterms: FOB Shanghai

Our annual volume is 100,000+ units across all categories.

www.modamilano.it"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_011", "纺织服装", "北美", "美国", "零售商/终端用户", "标准询盘",
    """分析这个美国客户的询盘:

Hi there,
We are FitGear Athletics, a fast-growing athleisure brand in Los Angeles.
Looking for a manufacturer for our leggings line.

Specs:
- Women's high-waist yoga leggings
- Fabric: Nylon 80% + Spandex 20%, 4-way stretch
- Weight: 200-220gsm, moisture-wicking
- Compression: medium
- Butt scrunch seam detail
- Custom logo waistband elastic
- Colors: Black, Navy, Wine Red, Sage Green

Order qty: 2000 pcs per color (total 8000 pcs) for first drop
Size range: XS-XL inclusive
Need GRS/OEKO-TEX certification for fabric

Target FOB price: under $7/pc
Lead time: 30-40 days

www.fitgearathletics.com"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_012", "纺织服装", "非洲", "南非", "分销商", "简单RFQ",
    """分析这个南非客户的询盘:

Hello,
We are Cape Textile Distributors in Cape Town, South Africa.
We supply workwear and uniforms to mining and construction companies.

Please quote on:
- Cotton twill work trousers, navy blue, 280gsm
- Quantity: 5000 pairs
- Size range: 30-46 waist
- Need reflective tape on pockets (SABS standard)
- FOB price and best delivery time to Durban Port

We buy this item regularly, about 20,000 pairs per year.
Please send your best price for ongoing business.

Regards,
David"""
))

# ============================================================
# 5. 化工材料 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_013", "化工材料", "东南亚", "泰国", "品牌商", "标准询盘",
    """分析这个泰国客户的询盘:

Dear Supplier,

We are Thai Polymer Solutions Co., Ltd, a manufacturer of automotive plastic parts
in Rayong, Thailand (Eastern Seaboard Industrial Estate).

We need a stable supplier for:
- ABS Resin (Acrylonitrile Butadiene Styrene), General Purpose Grade
- Melt Flow Index: 20-30 g/10min at 220°C/10kg
- Impact Strength: >20 KJ/m² (Izod notched)
- Natural color pellets

Annual requirement: 500-800 MT per year
First trial order: 1 × 20ft container (approximately 18-20 MT)
Packaging: 25kg bags on pallets
CIF Laem Chabang Port

Please provide:
- Technical data sheet (TDS) and MSDS
- COA (Certificate of Analysis) template
- REACH/RoHS compliance statement
- Lead time and payment terms

Do you have agent/distributor in Thailand?

www.thaipolymer.co.th"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_014", "化工材料", "西欧", "德国", "品牌商", "复杂询盘",
    """分析这个德国客户的询盘:

Sehr geehrte Damen und Herren,

We are ChemTech Deutschland GmbH, a specialty chemical distributor based in Hamburg.
We supply water-based coating raw materials to the European automotive and furniture industries.

We are evaluating a new source for:
- Water-based Polyurethane Dispersion (PUD)
- Solid content: 35±1%
- pH: 7.0-9.0
- Application: wood coating, high gloss topcoat
- Must comply with: EU REACH regulation, German ChemVerbotsV
- No APEO, no formaldehyde, VOC < 50g/L

Annual potential: 200-300 MT
Initial order: 5 MT for lab and pilot testing, then scale up
Packaging: 200kg steel drums or 1000L IBC totes
Incoterms: CIF Hamburg

We require:
- Full REACH registration documentation
- Technical data sheet and application guide
- 3rd party test reports (SGS/TÜV preferred)
- Factory audit possibility (we can send our QA team)

www.chemtech-deutschland.de"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_015", "化工材料", "中东", "阿联酋", "分销商", "简单RFQ",
    """分析这个迪拜客户的询盘:

Dear Sir/Madam,

We are GulfChem Trading LLC, a chemical trading company in Dubai, UAE.

Please quote your best FOB/CIF prices for:
1. Titanium Dioxide (TiO2) Rutile R996 grade — 20 MT/month
2. Calcium Carbonate (CaCO3) ground, 1250 mesh — 50 MT/month
3. Lithopone B301 — 10 MT/month

All items for paint and coating industry in the Middle East market.
Packaging: 25kg bags or 1MT jumbo bags
Destination: Jebel Ali Port, Dubai

We have been in this business for 15 years and buy regularly.
If your prices are competitive, we can place monthly orders.

Please send your offer.
"""
))

# ============================================================
# 6. 家居用品 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_016", "家居用品", "北美", "美国", "品牌商", "复杂询盘",
    """分析这个美国客户的询盘:

Company: ModernNest Home, Portland OR
Contact: Emily Chen, Product Development Director

We are a DTC (direct-to-consumer) home organization brand selling on Amazon, Wayfair,
and our own Shopify store. We need a manufacturer for our new kitchen organization line.

Products:
A) Airtight Glass Food Storage Containers: Borosilicate glass, BPA-free lid with silicone seal
   - 3 sizes: 370ml / 640ml / 1040ml
   - Snap-lock lid mechanism
   - Microwave, oven, freezer, dishwasher safe
B) Bamboo Expandable Drawer Organizer
   - Natural Moso bamboo, food-grade mineral oil finish
   - Expandable width 33-56cm
   - Non-slip silicone feet

Quantities:
- Glass containers: 20,000 sets (3pc set) in first order
- Bamboo organizers: 10,000 pcs

Requirements:
- FDA food contact compliance for glass/lid/silicone
- LFGB or SGS test report
- FSC certification for bamboo products
- Retail-ready packaging: color box with barcode, inner carton, master carton
- Amazon FBA labeling and packaging requirements

Target FOB: under $5/set (glass) / under $4/pc (bamboo)
Payment: L/C at sight
Lead time: 45-60 days

Please send your existing catalog of kitchen organization products and any private label
experience with US retail channels.

www.modernnesthome.com"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_017", "家居用品", "大洋洲", "澳大利亚", "分销商", "标准询盘",
    """分析这个澳大利亚客户的询盘:

G'day,
We are Aussie Home Essentials Pty Ltd in Melbourne, Australia.
We import and distribute kitchenware to major retailers (Kmart, Big W, Target).

We're looking for:
- Stainless Steel Vacuum Insulated Water Bottles
- Double wall, 18/8 (304) stainless steel
- Sizes: 500ml, 750ml, 1000ml
- Powder coated exterior, copper lining inner
- BPA-free lid with carry loop
- Keeps cold 24hrs / hot 12hrs

First order: 30,000 pcs (mix of sizes and colors)
Annual projection: 200,000+ pcs across multiple orders

Requirements:
- LFGB or EU food grade test reports
- Must pass our third-party inspection (we use SGS Australia)
- Color box packaging with EAN barcode
- FOB pricing to Melbourne Port

www.aussiehomeessentials.com.au"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_018", "家居用品", "南美", "墨西哥", "零售商/终端用户", "简单RFQ",
    """分析这个墨西哥客户的询盘:

Hola,
Somos HogarMex SA de CV, cadena de tiendas de artículos para el hogar en Ciudad de México.

Necesitamos cotización para:
- Organizadores de plástico para cajones (drawer organizers)
- Material: PS (poliestireno) transparente
- Juego de 6 piezas (diferentes tamaños)
- Empaque: caja de color con instrucciones en español

Cantidad: 5000 juegos para primer pedido
Puerto: CIF Manzanillo, México
Certificación: NOM (Norma Oficial Mexicana) requerida

Por favor envíen su mejor precio y catálogo.

www.hogarmex.mx"""
))

# ============================================================
# 7. 汽车配件 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_019", "汽车配件", "西欧", "德国", "品牌商", "复杂询盘",
    """分析这个德国客户的询盘:

Company: AutoTeile GmbH, Stuttgart, Germany
Contact: Markus Weber, Senior Procurement Manager

We are a Tier-2 automotive parts supplier to BMW, Mercedes-Benz, and VW Group.
We are looking for a new supplier of precision machined components.

Part: Brake Caliper Guide Pins
- Material: SUS304/316 stainless steel
- Machining: CNC turning + centerless grinding
- Surface finish: Ra ≤ 0.4μm
- Tolerance: ±0.01mm on diameter, ±0.05mm on length
- Heat treatment: induction hardening to HRC 50-55

Annual volume: 2,000,000 pcs across 6 part numbers
Initial order: 50,000 pcs (mixed part numbers) for PPAP qualification

Requirements:
- IATF 16949:2016 certified
- ISO 14001 environmental management
- PPAP Level 3 submission
- 100% dimensional inspection with CMM report
- Annual capacity verification audit

Incoterms: FOB Shanghai
Payment: 60 days net from invoice date
Supply agreement: 3-year framework contract

Please provide your quality certificates and automotive customer list.

www.autoteile-gmbh.de"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_020", "汽车配件", "北美", "美国", "分销商", "标准询盘",
    """分析这个美国客户的询盘:

Hi,
We are American Auto Parts Wholesale Inc., based in Houston TX.
We supply aftermarket auto parts to repair shops and online retailers across the US.

Looking for:
- Ceramic Brake Pads for popular US models:
  • Toyota Camry 2018-2025 (Front + Rear sets)
  • Honda Accord 2018-2025 (Front + Rear sets)
  • Ford F-150 2015-2025 (Front sets)

- Must meet FMVSS/SAE standards
- Copper-free formulation (CA/WA compliant)
- Packaging: our brand "DriveRight" in English/Spanish bilingual

First order: 2000 sets (mixed models)
Monthly volume: 500-1000 sets per month ongoing

Need FOB Qingdao price and lead time. Do you have existing US warehouse stock?

www.americanautopartswholesale.com"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_021", "汽车配件", "非洲", "尼日利亚", "分销商", "简单RFQ",
    """分析这个尼日利亚客户的询盘:

Dear Sir,

We are Lagos Auto Parts Ltd, a leading auto spare parts distributor in Nigeria
with shops in Ladipo Market, Lagos.

Please quote us for:
- Toyota Corolla 2010-2018 shock absorbers (front and rear)
- KYB or equivalent quality, Japanese technology preferred
- Quantity: 1000 pairs (500 front + 500 rear)
- CIF Apapa Port, Lagos

Also interested in:
- Brake discs (front) for Toyota Hilux 2015-2022
- Quantity: 500 pcs

SONCAP certification required for Nigeria customs clearance.
Please send your best CIF price.

Regards,
Emmanuel Okafor"""
))

# ============================================================
# 8. 医疗器械 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_022", "医疗器械", "西欧", "英国", "品牌商", "复杂询盘",
    """分析这个英国客户的询盘:

Company: MedSupply UK Ltd, London, United Kingdom
Contact: Dr. Sarah Thompson, Regulatory Affairs Director

We are a medical device distributor supplying NHS Trusts and private hospitals
across the UK and Ireland. We need a manufacturer for our own-brand PPE line.

Product 1: Type IIR Surgical Face Masks
- 3-ply non-woven, meltblown filter
- BFE ≥ 98%, differential pressure < 60 Pa/cm²
- Splash resistance ≥ 16.0 kPa
- Ear loop style
- EN 14683:2019 Type IIR compliant

Product 2: Nitrile Examination Gloves
- Powder-free, non-sterile
- AQL 1.5 (medical grade)
- Finger textured
- EN 455 Part 1-4 compliant

Annual volume: 50 million masks + 100 million gloves
Initial trial: 500,000 masks + 1,000,000 gloves

Critical requirements:
- UKCA / CE marking under EU MDR 2017/745
- ISO 13485:2016 certified facility
- FDA 510(k) preferred but not mandatory
- Full technical file available for UK MHRA registration
- Cleanroom Class 100,000 (ISO 8) minimum

We will audit your factory before placing orders.

www.medsupplyuk.co.uk"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_023", "医疗器械", "南美", "巴西", "分销商", "标准询盘",
    """分析这个巴西客户的询盘:

Prezado Fornecedor,

A MedBrasil Importações é uma importadora de dispositivos médicos localizada em São Paulo.
Estamos buscando um fabricante para:

- Luvas cirúrgicas estéreis de látex
- Tamanhos: 6.5, 7.0, 7.5, 8.0
- Superfície microtexturizada
- Embalagem individual (1 par/pacote)
- Esterilizadas por radiação gama

Quantidade inicial: 200,000 pares
Registro ANVISA necessário — precisamos de documentação completa

Favor enviar:
- Certificado ISO 13485
- Relatório de teste de biocompatibilidade (ISO 10993)
- Preço CIF Porto de Santos

www.medbrasil.com.br"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_024", "医疗器械", "东南亚", "印度尼西亚", "零售商/终端用户", "简单RFQ",
    """分析这个印尼客户的询盘:

Kepada Yth. Supplier,

Kami dari Medika Sejahtera, distributor alat kesehatan di Jakarta, Indonesia.
Kami membutuhkan:

- Tensimeter digital (digital blood pressure monitor)
- Tipe: Upper arm, fully automatic
- Layar LCD besar dengan backlight
- Fitur: deteksi detak jantung tidak teratur, memori 90 pembacaan
- Powered by: 4×AAA baterai atau USB-C
- Sertifikasi: KEMENKES RI (Kementerian Kesehatan Republik Indonesia)

Kuantitas: 3000 unit untuk pesanan percobaan
Harga: mohon penawaran FOB terbaik
Pengiriman: dalam 30 hari

Jika kualitas bagus, kami akan pesan rutin setiap 3 bulan.

www.medikasejahtera.co.id"""
))

# ============================================================
# 9. 太阳能 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_025", "太阳能", "西欧", "荷兰", "品牌商", "复杂询盘",
    """分析这个荷兰客户的询盘:

Company: SolarFlex Europe BV, Rotterdam, Netherlands
Contact: Lars van der Meer, CTO

We are a fast-growing solar energy solutions provider in the Benelux region.
We are launching our own brand of residential energy storage systems and need a partner.

Products:
A) LiFePO4 Wall-Mounted Battery Pack
   - Capacity: 5kWh and 10kWh modules, stackable up to 40kWh
   - Voltage: 48V (15S configuration, LiFePO4 prismatic cells)
   - Cycle life: ≥6000 cycles at 80% DoD
   - Communication: CAN/RS485, compatible with Victron, SMA, Growatt inverters
   - IP65, indoor/outdoor installation
   - UN38.3, IEC 62619, CE, UKCA certified

B) Hybrid Inverter
   - 5kW and 8kW single-phase models
   - PV input: up to 600V DC
   - Battery voltage: 48V
   - Peak efficiency ≥97.5%
   - VDE-AR-N 4105, EN 50549 compliant

Initial order: 100 battery units + 50 inverters
Annual target: 2000+ units

Critical: EU WEEE directive compliance and battery recycling program registration.
Please provide IEC test reports and EU Declaration of Conformity.

www.solarflex.eu"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_026", "太阳能", "非洲", "南非", "分销商", "标准询盘",
    """分析这个南非客户的询盘:

Dear Supplier,

We are Solar Africa Pty Ltd, a solar equipment distributor in Johannesburg, South Africa.
Due to the ongoing energy crisis (load shedding), demand for solar is booming here.

We need:
- Monocrystalline Solar Panels
- Power: 550W, half-cell, 144 cells
- Efficiency: ≥21.5%
- Frame: anodized aluminum alloy
- Connectors: MC4 compatible
- Certifications: IEC 61215, IEC 61730, SABS (South African Bureau of Standards)

Quantity: 2 × 40ft containers (approximately 1400 panels)
Monthly requirement: 3-5 containers ongoing
Port: CIF Durban

We need your best price and delivery schedule.
The South African market is very price-sensitive, so competitive pricing is critical.

www.solarafrica.co.za"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_027", "太阳能", "大洋洲", "澳大利亚", "零售商/终端用户", "标准询盘",
    """分析这个澳大利亚客户的询盘:

G'day mate,

We are SunRun Solar, an accredited solar installer based in Brisbane, QLD.
We install residential solar systems across Queensland and NSW.

We need:
- Solar Panels: 415W/420W N-type TOPCon, all-black, 108 half-cells
- Must be CEC (Clean Energy Council) approved for Australia
- IEC 61215 / IEC 61730 certified

- Hybrid Inverter: 5kW single-phase, 48V battery ready
- Must be CEC approved, AS/NZS 4777.2 compliant

Annual volume: 5-8MW of panels, 300-500 inverters
First order: 500 panels + 50 inverters for warehouse stock
FOB to Port of Brisbane

Critical: all products must be on the current CEC Approved Products List.

www.sunrun-solar.com.au"""
))

# ============================================================
# 10. 五金工具 (3 个)
# ============================================================

SEED_SCENARIOS.append(_make_inquiry(
    "seed_028", "五金工具", "西欧", "英国", "品牌商", "标准询盘",
    """分析这个英国客户的询盘:

Company: ProTool UK Ltd, Manchester, United Kingdom
Contact: James Wilson, Buying Manager

We are a professional tool brand selling to trade merchants and builders' merchants
across the UK (Screwfix, Toolstation, Travis Perkins).

We are looking to expand our cordless power tool range:

Cordless Impact Driver:
- 20V brushless motor, max torque ≥180Nm
- 1/4" hex chuck, 3-speed + variable speed trigger
- LED work light, belt clip
- Bare unit (no battery/charger) — must be compatible with our existing battery platform
- UKCA/CE marked

First order: 2000 units (bare tool only)
Target FOB: under $30/unit
Delivery: within 45 days

We also want:
- Cordless Angle Grinder 115mm/125mm 20V brushless — 1500 units
- Cordless Circular Saw 165mm 20V brushless — 1000 units

All tools must be compatible with our ProTool 20V MAX battery system.
Please send your OEM capabilities and existing cordless tool catalog.

www.protool-uk.co.uk"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_029", "五金工具", "中东", "沙特阿拉伯", "分销商", "标准询盘",
    """分析这个沙特客户的询盘:

Dear Supplier,

We are Arabian Tools Trading Est., a hardware and tools wholesaler in Riyadh,
Saudi Arabia. We supply construction tools to the massive NEOM and Red Sea mega projects.

We urgently need:
- SDS Plus Hammer Drill Bits Set (5, 6, 8, 10, 12, 14mm × 210mm working length)
- Material: carbide tip, hardened steel body
- German or Japanese quality level (Bosch/Hilti equivalent)
- Packed in professional plastic case

- Diamond Cutting Discs for angle grinder
- 115mm and 230mm, segmented turbo rim
- For granite, marble, and reinforced concrete

Quantities:
- Drill bit sets: 10,000 sets
- Diamond discs: 20,000 pcs (mixed sizes)

SASO/SABER certification required
CIF Dammam Islamic Port

This is for government infrastructure projects — quality and reliability are top priority.
Please send your best offer with full specifications.

www.arabiantools.com"""
))

SEED_SCENARIOS.append(_make_inquiry(
    "seed_030", "五金工具", "东南亚", "越南", "零售商/终端用户", "简单RFQ",
    """分析这个越南客户的询盘:

Chào nhà cung cấp,

Công ty chúng tôi là AnPhát Hardware, nhà bán lẻ dụng cụ cầm tay tại TP. Hồ Chí Minh.

Chúng tôi cần báo giá cho:
- Bộ cờ lê (combination wrench set) 25 chi tiết, 6-32mm
- Chất liệu: thép Cr-V (Chrome Vanadium)
- Bề mặt: đánh bóng gương (mirror polished)
- Tiêu chuẩn: DIN hoặc ANSI
- Đóng gói: hộp nhựa hoặc túi vải

Số lượng: 3000 bộ
Thời gian giao hàng: 30 ngày
Cảng: CIF Cát Lái, TP. Hồ Chí Minh

Chúng tôi nhập hàng đều đặn mỗi quý.
Vui lòng gửi báo giá và catalogue sản phẩm của quý công ty.

www.anphathardware.vn"""
))

# ============================================================
# 统计信息
# ============================================================

if __name__ == "__main__":
    print(f"种子场景总数: {len(SEED_SCENARIOS)}")
    print()
    print("行业分布:")
    from collections import Counter
    industries = Counter(s["industry"] for s in SEED_SCENARIOS)
    for ind, count in industries.most_common():
        print(f"  {ind}: {count}")
    print()
    print("地区分布:")
    regions = Counter(s["region"] for s in SEED_SCENARIOS)
    for reg, count in regions.most_common():
        print(f"  {reg}: {count}")
    print()
    print("买家类型分布:")
    buyers = Counter(s["buyer_type"] for s in SEED_SCENARIOS)
    for bt, count in buyers.most_common():
        print(f"  {bt}: {count}")
    print()
    print("复杂度分布:")
    compl = Counter(s["complexity"] for s in SEED_SCENARIOS)
    for c, count in compl.most_common():
        print(f"  {c}: {count}")
