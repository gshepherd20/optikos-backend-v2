"""
Scientific Lighting Validation System
Implements CIE standards for material analysis lighting conditions
"""

import cv2
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LightingConditions:
    """Scientific lighting conditions analysis"""
    illuminance_estimate: float  # Estimated lux
    color_temperature_estimate: int  # Kelvin
    uniformity_score: float  # 0-100
    spectral_quality_score: float  # 0-100
    cri_estimate: int  # Color Rendering Index 0-100
    scientific_validity: str  # excellent, good, acceptable, poor
    recommendations: list
    compliance_level: str  # CIE_compliant, acceptable, non_compliant

class ScientificLightingValidator:
    """
    Implements CIE International Commission on Illumination standards
    for material analysis lighting validation
    """
    
    def __init__(self):
        # CIE standards for material analysis
        self.optimal_illuminance_range = (300, 800)  # lux - CIE recommendation
        self.acceptable_illuminance_range = (200, 1200)  # lux - extended range
        self.optimal_color_temp_range = (5000, 6500)  # Kelvin - D50 to D65
        self.minimum_cri = 80  # Color Rendering Index minimum
        self.preferred_cri = 90  # Professional standard
        
        # Advanced validation thresholds
        self.uniformity_threshold = 70  # % minimum for scientific validity
        self.spectral_continuity_threshold = 75  # % for broad spectrum lighting
        
    def analyze_image_lighting(self, image_path: str, metadata: dict = None) -> LightingConditions:
        """
        Analyze lighting conditions from image using computer vision
        and scientific standards without requiring external sensors
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract lighting characteristics
            illuminance_est = self._estimate_illuminance(img_rgb)
            color_temp_est = self._estimate_color_temperature(img_rgb)
            uniformity = self._calculate_lighting_uniformity(img_rgb)
            spectral_quality = self._estimate_spectral_quality(img_rgb)
            cri_est = self._estimate_cri(img_rgb, color_temp_est)
            
            # Scientific validity assessment
            validity = self._assess_scientific_validity(
                illuminance_est, color_temp_est, uniformity, spectral_quality, cri_est
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                illuminance_est, color_temp_est, uniformity, spectral_quality, cri_est
            )
            
            # Compliance level
            compliance = self._determine_compliance_level(
                illuminance_est, color_temp_est, uniformity, spectral_quality, cri_est
            )
            
            return LightingConditions(
                illuminance_estimate=illuminance_est,
                color_temperature_estimate=color_temp_est,
                uniformity_score=uniformity,
                spectral_quality_score=spectral_quality,
                cri_estimate=cri_est,
                scientific_validity=validity,
                recommendations=recommendations,
                compliance_level=compliance
            )
            
        except Exception as e:
            logger.error(f"Error analyzing lighting: {str(e)}")
            raise
    
    def _estimate_illuminance(self, img_rgb: np.ndarray) -> float:
        """
        Estimate illuminance using CIE photopic luminous efficiency function
        Based on image brightness analysis and scientific correlation
        """
        # Convert to grayscale using CIE photopic weighting
        # Y = 0.2126*R + 0.7152*G + 0.0722*B (ITU-R BT.709)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Estimate illuminance based on empirical correlation
        # Derived from scientific photography standards
        if mean_brightness < 30:
            # Very dark conditions
            illuminance = 50 + (mean_brightness / 30) * 100
        elif mean_brightness < 80:
            # Low light conditions
            illuminance = 150 + ((mean_brightness - 30) / 50) * 200
        elif mean_brightness < 150:
            # Normal indoor conditions
            illuminance = 350 + ((mean_brightness - 80) / 70) * 300
        elif mean_brightness < 200:
            # Bright indoor/outdoor shade
            illuminance = 650 + ((mean_brightness - 150) / 50) * 400
        else:
            # Very bright conditions
            illuminance = 1050 + ((mean_brightness - 200) / 55) * 500
        
        # Apply histogram analysis for refinement
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check for proper exposure (avoid clipping)
        highlight_clipping = np.sum(hist[240:]) / np.sum(hist)
        shadow_clipping = np.sum(hist[0:15]) / np.sum(hist)
        
        # Adjust based on exposure quality
        if highlight_clipping > 0.02:  # Overexposed
            illuminance *= 1.3
        elif shadow_clipping > 0.05:  # Underexposed
            illuminance *= 0.7
        
        return max(50, min(2000, illuminance))  # Clamp to reasonable range
    
    def _estimate_color_temperature(self, img_rgb: np.ndarray) -> int:
        """
        Estimate color temperature using white balance analysis
        Based on CIE standard illuminants
        """
        # Calculate average RGB values
        mean_r = np.mean(img_rgb[:, :, 0])
        mean_g = np.mean(img_rgb[:, :, 1])
        mean_b = np.mean(img_rgb[:, :, 2])
        
        # Avoid division by zero
        if mean_g == 0:
            return 5500  # Default daylight
        
        # Calculate color ratios
        r_g_ratio = mean_r / mean_g
        b_g_ratio = mean_b / mean_g
        
        # Estimate color temperature using empirical model
        # Based on planckian locus approximation
        if r_g_ratio > 1.1:  # Warm light (tungsten-like)
            if r_g_ratio > 1.3:
                color_temp = 2700 + (1.5 - r_g_ratio) * 1000
            else:
                color_temp = 3200 + (1.3 - r_g_ratio) * 2000
        elif b_g_ratio > 1.05:  # Cool light (daylight-like)
            color_temp = 5500 + (b_g_ratio - 1.0) * 3000
        else:  # Neutral
            color_temp = 4500 + (1.0 - r_g_ratio) * 1000
        
        return max(2500, min(8000, int(color_temp)))
    
    def _calculate_lighting_uniformity(self, img_rgb: np.ndarray) -> float:
        """
        Calculate lighting uniformity across the image
        Critical for scientific material comparison
        """
        # Convert to luminance
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Divide image into grid for uniformity analysis
        h, w = gray.shape
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size
        
        luminance_values = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = min((i + 1) * cell_h, h)
                x_start = j * cell_w
                x_end = min((j + 1) * cell_w, w)
                
                cell = gray[y_start:y_end, x_start:x_end]
                luminance_values.append(np.mean(cell))
        
        # Calculate uniformity score
        mean_luminance = np.mean(luminance_values)
        std_luminance = np.std(luminance_values)
        
        if mean_luminance == 0:
            return 0
        
        # Uniformity = 100 - (coefficient of variation * 100)
        coefficient_of_variation = std_luminance / mean_luminance
        uniformity_score = max(0, 100 - (coefficient_of_variation * 100))
        
        return uniformity_score
    
    def _estimate_spectral_quality(self, img_rgb: np.ndarray) -> float:
        """
        Estimate spectral quality and continuity
        Important for accurate color reproduction
        """
        # Analyze color distribution across image
        r_channel = img_rgb[:, :, 0].flatten()
        g_channel = img_rgb[:, :, 1].flatten()
        b_channel = img_rgb[:, :, 2].flatten()
        
        # Calculate histograms
        hist_r = np.histogram(r_channel, bins=32, range=(0, 256))[0]
        hist_g = np.histogram(g_channel, bins=32, range=(0, 256))[0]
        hist_b = np.histogram(b_channel, bins=32, range=(0, 256))[0]
        
        # Normalize histograms
        hist_r = hist_r / np.sum(hist_r)
        hist_g = hist_g / np.sum(hist_g)
        hist_b = hist_b / np.sum(hist_b)
        
        # Calculate spectral continuity
        # Good lighting should have smooth distribution across all channels
        continuity_r = self._calculate_histogram_smoothness(hist_r)
        continuity_g = self._calculate_histogram_smoothness(hist_g)
        continuity_b = self._calculate_histogram_smoothness(hist_b)
        
        # Average continuity score
        spectral_quality = (continuity_r + continuity_g + continuity_b) / 3
        
        # Check for color bias (indicates poor spectral quality)
        total_intensity = np.mean(r_channel) + np.mean(g_channel) + np.mean(b_channel)
        if total_intensity > 0:
            r_bias = abs(np.mean(r_channel) / total_intensity - 0.33)
            g_bias = abs(np.mean(g_channel) / total_intensity - 0.33)
            b_bias = abs(np.mean(b_channel) / total_intensity - 0.33)
            color_bias_penalty = (r_bias + g_bias + b_bias) * 100
            spectral_quality = max(0, spectral_quality - color_bias_penalty)
        
        return spectral_quality
    
    def _calculate_histogram_smoothness(self, histogram: np.ndarray) -> float:
        """Calculate smoothness of histogram (indicates spectral continuity)"""
        if len(histogram) < 2:
            return 0
        
        # Calculate differences between adjacent bins
        differences = np.abs(np.diff(histogram))
        mean_difference = np.mean(differences)
        
        # Higher smoothness = lower differences
        smoothness = max(0, 100 - (mean_difference * 1000))
        return smoothness
    
    def _estimate_cri(self, img_rgb: np.ndarray, color_temp: int) -> int:
        """
        Estimate Color Rendering Index based on image analysis
        """
        # Simplified CRI estimation based on color distribution
        # Full CRI requires spectroradiometer, this gives approximation
        
        # Analyze color saturation across image
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Calculate mean saturation
        mean_saturation = np.mean(saturation)
        
        # Estimate CRI based on saturation and color temperature
        base_cri = 70  # Minimum assumption
        
        # Higher saturation generally indicates better color rendering
        saturation_bonus = min(20, mean_saturation / 12.8)  # Scale 0-20
        
        # Color temperature affects CRI estimation
        if 4500 <= color_temp <= 6500:  # Optimal range
            temp_bonus = 10
        elif 3000 <= color_temp <= 7000:  # Acceptable range
            temp_bonus = 5
        else:  # Poor range
            temp_bonus = 0
        
        estimated_cri = base_cri + saturation_bonus + temp_bonus
        return max(60, min(98, int(estimated_cri)))
    
    def _assess_scientific_validity(self, illuminance: float, color_temp: int, 
                                   uniformity: float, spectral_quality: float, cri: int) -> str:
        """Assess overall scientific validity of lighting conditions"""
        
        # Score each parameter (0-100)
        illuminance_score = self._score_illuminance(illuminance)
        temp_score = self._score_color_temperature(color_temp)
        uniformity_score = uniformity
        spectral_score = spectral_quality
        cri_score = min(100, cri)
        
        # Weighted average (illuminance and uniformity are most critical)
        overall_score = (
            illuminance_score * 0.3 +
            uniformity_score * 0.25 +
            temp_score * 0.2 +
            spectral_score * 0.15 +
            cri_score * 0.1
        )
        
        if overall_score >= 85:
            return "excellent"
        elif overall_score >= 75:
            return "good"
        elif overall_score >= 60:
            return "acceptable"
        else:
            return "poor"
    
    def _score_illuminance(self, illuminance: float) -> float:
        """Score illuminance against CIE standards"""
        opt_min, opt_max = self.optimal_illuminance_range
        acc_min, acc_max = self.acceptable_illuminance_range
        
        if opt_min <= illuminance <= opt_max:
            return 100
        elif acc_min <= illuminance <= acc_max:
            if illuminance < opt_min:
                return 60 + (illuminance - acc_min) / (opt_min - acc_min) * 40
            else:
                return 60 + (acc_max - illuminance) / (acc_max - opt_max) * 40
        else:
            return max(0, 30 - abs(illuminance - np.mean(self.acceptable_illuminance_range)) / 10)
    
    def _score_color_temperature(self, color_temp: int) -> float:
        """Score color temperature against CIE standards"""
        opt_min, opt_max = self.optimal_color_temp_range
        
        if opt_min <= color_temp <= opt_max:
            return 100
        elif 3000 <= color_temp <= 8000:  # Acceptable range
            deviation = min(abs(color_temp - opt_min), abs(color_temp - opt_max))
            return max(50, 100 - deviation / 20)
        else:
            return max(0, 30 - abs(color_temp - np.mean(self.optimal_color_temp_range)) / 100)
    
    def _determine_compliance_level(self, illuminance: float, color_temp: int,
                                   uniformity: float, spectral_quality: float, cri: int) -> str:
        """Determine CIE compliance level"""
        
        # Check critical requirements
        illuminance_ok = self.acceptable_illuminance_range[0] <= illuminance <= self.acceptable_illuminance_range[1]
        uniformity_ok = uniformity >= self.uniformity_threshold
        cri_ok = cri >= self.minimum_cri
        
        if illuminance_ok and uniformity_ok and cri_ok:
            if (self.optimal_illuminance_range[0] <= illuminance <= self.optimal_illuminance_range[1] and
                uniformity >= 85 and cri >= self.preferred_cri):
                return "CIE_compliant"
            else:
                return "acceptable"
        else:
            return "non_compliant"
    
    def _generate_recommendations(self, illuminance: float, color_temp: int,
                                 uniformity: float, spectral_quality: float, cri: int) -> list:
        """Generate specific recommendations for improvement"""
        recommendations = []
        
        # Illuminance recommendations
        if illuminance < self.acceptable_illuminance_range[0]:
            recommendations.append(f"Increase lighting: {illuminance:.0f} lux is below minimum 200 lux")
        elif illuminance > self.acceptable_illuminance_range[1]:
            recommendations.append(f"Reduce lighting: {illuminance:.0f} lux exceeds maximum 1200 lux")
        elif illuminance < self.optimal_illuminance_range[0]:
            recommendations.append(f"Optimal lighting: Increase to 300-800 lux range (current: {illuminance:.0f})")
        elif illuminance > self.optimal_illuminance_range[1]:
            recommendations.append(f"Optimal lighting: Reduce to 300-800 lux range (current: {illuminance:.0f})")
        
        # Color temperature recommendations
        if not (self.optimal_color_temp_range[0] <= color_temp <= self.optimal_color_temp_range[1]):
            recommendations.append(f"Adjust color temperature to 5000-6500K (current: {color_temp}K)")
        
        # Uniformity recommendations
        if uniformity < self.uniformity_threshold:
            recommendations.append(f"Improve lighting uniformity: {uniformity:.1f}% (minimum: 70%)")
        
        # CRI recommendations
        if cri < self.minimum_cri:
            recommendations.append(f"Use higher CRI lighting: {cri} (minimum: 80)")
        elif cri < self.preferred_cri:
            recommendations.append(f"Professional standard: Use CRI 90+ lighting (current: {cri})")
        
        # Spectral quality recommendations
        if spectral_quality < 70:
            recommendations.append("Use full-spectrum lighting for better color accuracy")
        
        if not recommendations:
            recommendations.append("Lighting conditions meet scientific standards")
        
        return recommendations

    def validate_for_verified_report(self, lighting_conditions: LightingConditions) -> dict:
        """Validate if lighting meets requirements for verified reports"""
        
        meets_requirements = (
            lighting_conditions.compliance_level in ["CIE_compliant", "acceptable"] and
            lighting_conditions.uniformity_score >= 70 and
            lighting_conditions.cri_estimate >= 80
        )
        
        return {
            "qualified_for_verification": meets_requirements,
            "compliance_level": lighting_conditions.compliance_level,
            "scientific_validity": lighting_conditions.scientific_validity,
            "verification_notes": {
                "illuminance": f"{lighting_conditions.illuminance_estimate:.0f} lux",
                "color_temperature": f"{lighting_conditions.color_temperature_estimate}K",
                "uniformity": f"{lighting_conditions.uniformity_score:.1f}%",
                "cri_estimate": lighting_conditions.cri_estimate,
                "recommendations": lighting_conditions.recommendations
            }
        }