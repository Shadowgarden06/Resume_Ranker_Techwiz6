# src/feedback_system.py
import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Optional
from flask import session
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

class FeedbackSystem:
    """Feedback system using CSV to improve ranking model accuracy"""
    
    def __init__(self, data_dir: str = "feedback_data"):
        self.data_dir = data_dir
        self.feedback_file = os.path.join(data_dir, "feedback.csv")
        self.weights_file = os.path.join(data_dir, "model_weights.json")
        
        # Create directory if not exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize CSV file if not exists
        self.init_csv_file()
        
        # Load current weights
        self.current_weights = self.load_current_weights()
    
    def init_csv_file(self):
        """Initialize CSV file with header if not exists"""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'session_id', 'user_id', 'job_description_hash',
                    'candidate_filename', 'candidate_name', 'candidate_email',
                    'ai_rank', 'ai_combined_score', 'ai_tfidf_score', 
                    'ai_semantic_score', 'ai_skill_score',
                    'human_decision', 'human_rating', 'feedback_notes'
                ])
            print(f"✅ Created feedback CSV file: {self.feedback_file}")
    
    def load_current_weights(self) -> Dict[str, float]:
        """Load current weights from JSON file"""
        default_weights = {
            'tfidf': 0.4,
            'semantic': 0.5,
            'skill': 0.1
        }
        
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r', encoding='utf-8') as f:
                    weights = json.load(f)
                    print(f"📊 Loaded weights: {weights}")
                    return weights
            except Exception as e:
                print(f"⚠️ Error loading weights, using default: {e}")
        
        return default_weights
    
    def save_weights(self, weights: Dict[str, float]) -> bool:
        """Save new weights to JSON file"""
        try:
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights, f, ensure_ascii=False, indent=2)
            
            self.current_weights = weights
            print(f"💾 Saved new weights: {weights}")
            return True
        except Exception as e:
            print(f"❌ Error saving weights: {e}")
            return False
    
    def save_feedback(self, 
                     candidate_filename: str,
                     candidate_name: str,
                     candidate_email: str,
                     ai_rank: int,
                     ai_scores: Dict[str, float],
                     human_decision: str,
                     human_rating: Optional[int] = None,
                     feedback_notes: Optional[str] = None) -> bool:
        """Save feedback from recruiter to CSV"""
        try:
            # Create session info
            session_id = session.get('session_id', f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            user_id = session.get('user_id', 'anonymous')
            job_description = session.get('job_description', '')
            
            # Create hash of job description to avoid saving too much text
            job_description_hash = str(hash(job_description))[:10]
            
            # Prepare data
            row_data = [
                datetime.now().isoformat(),
                session_id,
                user_id,
                job_description_hash,
                candidate_filename,
                candidate_name,
                candidate_email,
                ai_rank,
                ai_scores['combined'],
                ai_scores['tfidf'],
                ai_scores['semantic'],
                ai_scores['skill'],
                human_decision,
                human_rating or '',
                feedback_notes or ''
            ]
            
            # Write to CSV
            with open(self.feedback_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
            
            print(f"✅ Saved feedback CSV: {candidate_filename} - {human_decision}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving feedback CSV: {e}")
            return False
    
    def get_feedback_history(self, limit: int = 100) -> List[Dict]:
        """Get feedback history from CSV"""
        try:
            if not os.path.exists(self.feedback_file):
                return []
            
            df = pd.read_csv(self.feedback_file)
            
            # Sort by timestamp descending
            df = df.sort_values('timestamp', ascending=False)
            
            # Limit quantity
            df = df.head(limit)
            
            # Convert to list of dicts
            results = df.to_dict('records')
            
            print(f"📊 Loaded {len(results)} feedback records")
            return results
            
        except Exception as e:
            print(f"❌ Error reading feedback CSV: {e}")
            return []
    
    def get_feedback_statistics(self) -> Dict:
        """Get feedback statistics from CSV"""
        try:
            if not os.path.exists(self.feedback_file):
                return {
                    'total_feedback': 0,
                    'decision_distribution': {},
                    'average_rating': 0,
                    'recent_feedback': 0
                }
            
            df = pd.read_csv(self.feedback_file)
            
            # Total feedback count
            total_feedback = len(df)
            
            # Decision distribution
            decision_distribution = df['human_decision'].value_counts().to_dict()
            
            # Average rating (only count records with rating)
            avg_rating = 0
            if 'human_rating' in df.columns:
                ratings = df['human_rating'].dropna()
                if len(ratings) > 0:
                    avg_rating = ratings.astype(float).mean()
            
            # Recent feedback (7 days)
            recent_feedback = 0
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                week_ago = datetime.now() - pd.Timedelta(days=7)
                recent_feedback = len(df[df['timestamp'] >= week_ago])
            
            return {
                'total_feedback': total_feedback,
                'decision_distribution': decision_distribution,
                'average_rating': round(avg_rating, 2),
                'recent_feedback': recent_feedback
            }
            
        except Exception as e:
            print(f"❌ Error calculating statistics: {e}")
            return {}
    
    def calculate_model_improvement(self) -> Dict[str, float]:
        """Calculate model improvement based on feedback CSV"""
        try:
            if not os.path.exists(self.feedback_file):
                return {"status": "no_data"}
            
            df = pd.read_csv(self.feedback_file)
            
            if len(df) < 10:  # Need at least 10 feedback
                return {"status": "insufficient_data", "message": f"Only {len(df)} feedback, need at least 10"}
            
            # Prepare data for machine learning
            # Only get records with human_rating
            df_with_rating = df[df['human_rating'].notna() & (df['human_rating'] != '')]
            
            if len(df_with_rating) < 5:
                return {"status": "insufficient_ratings", "message": f"Only {len(df_with_rating)} rating, need at least 5"}
            
            X = df_with_rating[['ai_tfidf_score', 'ai_semantic_score', 'ai_skill_score']].values
            y = []
            
            for _, row in df_with_rating.iterrows():
                # Convert human decision to number
                decision_score = self._convert_decision_to_score(row['human_decision'])
                
                # Combine rating and decision
                if pd.notna(row['human_rating']) and row['human_rating'] != '':
                    rating = float(row['human_rating'])
                    final_score = (decision_score + rating) / 2
                else:
                    final_score = decision_score
                
                y.append(final_score)
            
            y = np.array(y)
            
            if len(X) < 5:
                return {"status": "insufficient_data"}
            
            # Normalize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest to find feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            
            feature_importance = rf.feature_importances_
            
            # Normalize to weights
            total_importance = np.sum(feature_importance)
            new_weights = {
                'tfidf': float(feature_importance[0] / total_importance),
                'semantic': float(feature_importance[1] / total_importance), 
                'skill': float(feature_importance[2] / total_importance)
            }
            
            # Calculate improvement score
            improvement_score = rf.score(X_scaled, y)
            
            print(f"�� Model improvement calculated:")
            print(f"   - New weights: {new_weights}")
            print(f"   - Improvement score: {improvement_score:.3f}")
            print(f"   - Sample size: {len(X)}")
            
            return {
                "status": "success",
                "new_weights": new_weights,
                "improvement_score": improvement_score,
                "sample_size": len(X),
                "feature_importance": {
                    'tfidf': float(feature_importance[0]),
                    'semantic': float(feature_importance[1]),
                    'skill': float(feature_importance[2])
                }
            }
            
        except Exception as e:
            print(f"❌ Error calculating improvement: {e}")
            return {"status": "error", "message": str(e)}
    
    def _convert_decision_to_score(self, decision: str) -> float:
        """Convert human decision to score"""
        decision_scores = {
            'hired': 5.0,
            'interviewed': 4.0,
            'shortlisted': 3.0,
            'rejected': 1.0,
            'not_suitable': 1.0
        }
        return decision_scores.get(decision.lower(), 2.5)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.current_weights
    
    def export_feedback_to_csv(self, output_file: str = None) -> str:
        """Export feedback data to a new CSV file"""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.data_dir, f"feedback_export_{timestamp}.csv")
            
            if os.path.exists(self.feedback_file):
                df = pd.read_csv(self.feedback_file)
                df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"📤 Exported feedback to: {output_file}")
                return output_file
            else:
                print("❌ No feedback file to export")
                return None
                
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
            return None

class AutoRetrainSystem:
    """Automatic model retraining system"""
    
    def __init__(self, feedback_system):
        self.feedback_system = feedback_system
        self.last_retrain_time = None
        self.retrain_threshold = 10  # Minimum number of feedbacks to retrain
        
    def should_retrain(self) -> bool:
        """Check if should retrain"""
        try:
            stats = self.feedback_system.get_feedback_statistics()
            
            # Condition 1: Enough new feedback
            if stats['total_feedback'] >= self.retrain_threshold:
                return True
                
            # Condition 2: New feedback in 24h and total >= 5
            if stats['recent_feedback'] >= 3 and stats['total_feedback'] >= 5:
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error checking retrain condition: {e}")
            return False
    
    def auto_retrain(self) -> Dict:
        """Automatically retrain if conditions are met"""
        try:
            if not self.should_retrain():
                return {"status": "no_retrain_needed"}
            
            print("🔄 Starting auto retrain...")
            result = self.feedback_system.calculate_model_improvement()
            
            if result.get('status') == 'success':
                # Save new weights
                self.feedback_system.save_weights(result['new_weights'])
                self.last_retrain_time = datetime.now()
                
                print(f"✅ Auto retrain completed. New weights: {result['new_weights']}")
                return {
                    "status": "retrained",
                    "new_weights": result['new_weights'],
                    "improvement_score": result['improvement_score']
                }
            else:
                return {"status": "retrain_failed", "reason": result.get('message')}
                
        except Exception as e:
            print(f"❌ Error in auto retrain: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_retrain_status(self) -> Dict:
        """Get retraining status"""
        return {
            "should_retrain": self.should_retrain(),
            "last_retrain": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "threshold": self.retrain_threshold
        }

def calculate_model_improvement_advanced(self) -> Dict[str, float]:
    """Advanced model improvement calculation with multiple algorithms"""
    try:
        if not os.path.exists(self.feedback_file):
            return {"status": "no_data"}
        
        df = pd.read_csv(self.feedback_file)
        
        if len(df) < 10:
            return {"status": "insufficient_data", "message": f"Only {len(df)} feedback, need at least 10"}
        
        # Prepare data
        df_with_rating = df[df['human_rating'].notna() & (df['human_rating'] != '')]
        
        if len(df_with_rating) < 5:
            return {"status": "insufficient_ratings", "message": f"Only {len(df_with_rating)} ratings, need at least 5"}
        
        X = df_with_rating[['ai_tfidf_score', 'ai_semantic_score', 'ai_skill_score']].values
        y = []
        
        for _, row in df_with_rating.iterrows():
            decision_score = self._convert_decision_to_score(row['human_decision'])
            
            if pd.notna(row['human_rating']) and row['human_rating'] != '':
                rating = float(row['human_rating'])
                # Higher weight for rating than decision
                final_score = (0.7 * rating + 0.3 * decision_score)
            else:
                final_score = decision_score
            
            y.append(final_score)
        
        y = np.array(y)
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try multiple algorithms and choose the best
        algorithms = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        best_score = 0
        best_weights = None
        best_algorithm = None
        
        for name, model in algorithms.items():
            try:
                model.fit(X_scaled, y)
                score = model.score(X_scaled, y)
                
                if score > best_score:
                    best_score = score
                    best_algorithm = name
                    
                    # Get feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        # Linear regression coefficients
                        feature_importance = np.abs(model.coef_)
                        feature_importance = feature_importance / np.sum(feature_importance)
                    else:
                        # Default equal weights
                        feature_importance = np.array([1/3, 1/3, 1/3])
                    
                    total_importance = np.sum(feature_importance)
                    best_weights = {
                        'tfidf': float(feature_importance[0] / total_importance),
                        'semantic': float(feature_importance[1] / total_importance),
                        'skill': float(feature_importance[2] / total_importance)
                    }
                    
            except Exception as e:
                print(f"⚠️ Error with {name}: {e}")
                continue
        
        if best_weights is None:
            return {"status": "error", "message": "All algorithms failed"}
        
        print(f"🏆 Best algorithm: {best_algorithm} (score: {best_score:.3f})")
        print(f"📊 New weights: {best_weights}")
        
        return {
            "status": "success",
            "new_weights": best_weights,
            "improvement_score": best_score,
            "best_algorithm": best_algorithm,
            "sample_size": len(X),
            "feature_importance": {
                'tfidf': float(feature_importance[0]),
                'semantic': float(feature_importance[1]),
                'skill': float(feature_importance[2])
            }
        }
        
    except Exception as e:
        print(f"❌ Error in advanced model improvement: {e}")
        return {"status": "error", "message": str(e)}

def save_feedback_with_auto_retrain(self, 
                     candidate_filename: str,
                     candidate_name: str,
                     candidate_email: str,
                     ai_rank: int,
                     ai_scores: Dict[str, float],
                     human_decision: str,
                     human_rating: Optional[int] = None,
                     feedback_notes: Optional[str] = None) -> bool:
    """Save feedback and automatically retrain if needed"""
    try:
        # Save feedback as usual
        success = self.save_feedback(
            candidate_filename, candidate_name, candidate_email,
            ai_rank, ai_scores, human_decision, human_rating, feedback_notes
        )
        
        if success:
            # Check and automatically retrain
            if hasattr(self, 'auto_retrain_system'):
                retrain_result = self.auto_retrain_system.auto_retrain()
                if retrain_result.get('status') == 'retrained':
                    print(f"🤖 Auto retrained with new weights: {retrain_result['new_weights']}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error in save_feedback_with_auto_retrain: {e}")
        return False

# Global instance
feedback_system = FeedbackSystem()
auto_retrain_system = AutoRetrainSystem(feedback_system)
